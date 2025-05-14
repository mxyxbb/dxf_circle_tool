import ezdxf
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from shapely.geometry import Polygon, MultiPoint, LineString, box
import numpy as np
from shapely import concave_hull
import os
from matplotlib.colors import to_hex
import random
import tkinter as tk
from tkinter import filedialog
import math
from collections import defaultdict

# 设置中文字体和符号显示
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def select_file():
    """使用文件选择器选择DXF文件"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择DXF文件",
        filetypes=[("DXF文件", "*.dxf"), ("所有文件", "*.*")],
        initialdir=os.getcwd()
    )
    return file_path

def is_circle_from_lines(lines, tolerance=0.01):
    """判断LINE组是否构成圆形"""
    if len(lines) < 8:
        return False
    
    # 收集所有端点
    points = []
    for line in lines:
        points.append((line.dxf.start.x, line.dxf.start.y))
        points.append((line.dxf.end.x, line.dxf.end.y))
    
    # 计算中心点和半径
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    radii = [math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in points]
    avg_radius = sum(radii) / len(radii)
    radius_std = math.sqrt(sum((r - avg_radius)**2 for r in radii) / len(radii))
    
    return radius_std < tolerance * avg_radius

def remove_circular_entities(doc):
    """删除由LINE组成的圆形和CIRCLE类型的圆形"""
    msp = doc.modelspace()
    deleted_count = 0
    
    # 删除CIRCLE类型的圆形
    circles = list(msp.query('CIRCLE'))
    for circle in circles:
        msp.delete_entity(circle)
    deleted_count += len(circles)
    
    # 删除由LINE组成的圆形
    all_lines = list(msp.query('LINE'))
    line_groups = []
    used_lines = set()
    
    for i, line in enumerate(all_lines):
        if i in used_lines:
            continue
            
        current_group = [line]
        used_lines.add(i)
        last_point = (line.dxf.end.x, line.dxf.end.y)
        
        while True:
            found = False
            for j, other_line in enumerate(all_lines):
                if j in used_lines:
                    continue
                    
                start = (other_line.dxf.start.x, other_line.dxf.start.y)
                if (abs(start[0] - last_point[0]) < 1e-7 and 
                    abs(start[1] - last_point[1]) < 1e-7):
                    current_group.append(other_line)
                    used_lines.add(j)
                    last_point = (other_line.dxf.end.x, other_line.dxf.end.y)
                    found = True
                    break
                    
            if not found:
                break
        
        line_groups.append(current_group)
    
    # 删除圆形线段组
    for group in line_groups:
        if is_circle_from_lines(group):
            for line in group:
                msp.delete_entity(line)
            deleted_count += len(group)
    
    return deleted_count

class DxfPointManager:
    def __init__(self, filename=None, points=None):
        self.filename = filename
        self.original_doc = None
        self.original_circles = []
        self.selection_history = []
        self.polygon_history = []
        self.deleted_indices = set()  # 记录被删除的点
        
        if filename:
            self.load_dxf(filename)
        elif points is not None:
            self.original_points = np.array(points)
            self.available_indices = set(range(len(points)))
            self.used_indices = set()
        
        self.polygons = []
    
    def load_dxf(self, filename):
        """从DXF文件加载所有圆点"""
        try:
            self.original_doc = ezdxf.readfile(filename)
            self.filename = filename
            msp = self.original_doc.modelspace()
            
            points = []
            self.original_circles = []
            
            # 读取所有圆，不限制半径
            for circle in msp.query('CIRCLE'):
                center = circle.dxf.center
                # points.append([center.x, center.y, circle.dxf.radius])  # 包含半径信息
                points.append([round(center.x,4), round(center.y,4), round(circle.dxf.radius,4)])
                self.original_circles.append(circle)
            
            if not points:
                raise ValueError("DXF文件中未找到任何圆")
            
            # 转换为numpy数组并确保是二维的
            points = np.array(points)
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            
            # 按Y和X坐标排序
            y_sorted = sorted(points, key=itemgetter(1))
            xy_sorted = sorted(y_sorted, key=itemgetter(0))
            
            # 重新排列圆对象以匹配排序后的点
            sorted_circles = []
            used_indices = set()  # 记录已使用的圆索引
            
            for x, y, r in xy_sorted:
                for i, circle in enumerate(self.original_circles):
                    if i in used_indices:
                        continue
                    if (abs(circle.dxf.center.x - x) < 1e-7 and 
                        abs(circle.dxf.center.y - y) < 1e-7 and
                        abs(circle.dxf.radius - r) < 1e-7):
                        sorted_circles.append(circle)
                        used_indices.add(i)
                        break
            
            # 处理重合的圆 - 只保留一个
            unique_points = []
            unique_circles = []
            seen = set()
            
            for i, (x, y, r) in enumerate(xy_sorted):
                key = (round(x, 4), round(y, 4))  # 使用坐标作为唯一标识
                if key not in seen:
                    seen.add(key)
                    unique_points.append([x, y, r])
                    unique_circles.append(sorted_circles[i])
            
            self.original_circles = unique_circles
            self.original_points = np.array([[x, y] for x, y, r in unique_points])  # 只保存坐标
            self.available_indices = set(range(len(self.original_points)))
            self.used_indices = set()
            self.radii = np.array([r for x, y, r in unique_points])  # 保存半径
            
        except Exception as e:
            print(f"加载DXF文件失败: {e}")
            raise
    
    def add_selection(self, indices, polygon_data=None):
        self.selection_history.append({
            'indices': set(indices),
            'polygon_data': polygon_data
        })
    
    def pop_last_selection(self):
        if self.selection_history:
            return self.selection_history.pop()
        return None
    
    @property
    def available_points(self):
        """获取当前可用点的坐标（不包括已删除的点）"""
        active_indices = list(self.available_indices - self.deleted_indices)
        return self.original_points[active_indices]
    
    def get_active_indices(self):
        """获取当前活跃点的索引（不包括已删除的点）"""
        return list(self.available_indices - self.deleted_indices)
    
    def use_points(self, indices):
        """标记点已被使用"""
        indices_set = set(indices)
        self.used_indices.update(indices_set)
        self.available_indices -= indices_set
    
    def restore_points(self, indices):
        """恢复点"""
        indices_set = set(indices)
        self.used_indices -= indices_set
        self.available_indices.update(indices_set)
    
    def delete_points(self, indices):
        """删除点（不是用于多边形，而是永久删除）"""
        indices_set = set(indices)
        self.deleted_indices.update(indices_set)
    
    def restore_deleted_points(self, indices):
        """恢复被删除的点"""
        indices_set = set(indices)
        self.deleted_indices -= indices_set
    
    def get_used_points(self):
        return self.original_points[list(self.used_indices)]
    
    def add_polygon(self, polygon, polygon_type, color):
        polygon_data = {
            'polygon': polygon,
            'type': polygon_type,
            'color': color,
            'used_indices': list(self.used_indices)
        }
        self.polygons.append(polygon_data)
        return polygon_data
    
    def remove_last_polygon(self):
        if self.polygons:
            return self.polygons.pop()
        return None
    
    def save_to_dxf(self, save_path):
        if not self.original_doc:
            print("没有可保存的DXF文档")
            return False
        
        try:
            # 创建新文档
            doc = ezdxf.new('R2010')
            
            # 创建图层
            doc.layers.new("POLYGONS", dxfattribs={'color': 2})  # 绿色
            doc.layers.new("ORIGINAL_ENTITIES", dxfattribs={'color': 7})  # 白色
            
            msp = doc.modelspace()
            
            # 1. 保存原始DXF中的所有非圆图形
            original_msp = self.original_doc.modelspace()
            for entity in original_msp:
                try:
                    # 跳过圆和None值
                    if entity is None or entity.dxftype() in ['CIRCLE']:
                        continue
                        
                    # 复制实体到新文档
                    new_entity = entity.copy()
                    if new_entity is not None:
                        """在复制实体时增加坐标取整"""
                        if entity.dxftype() == 'LINE':
                            new_entity.dxf.start = (round(entity.dxf.start.x,4), round(entity.dxf.start.y,4))
                            new_entity.dxf.end = (round(entity.dxf.end.x,4), round(entity.dxf.end.y,4))
                        msp.add_entity(new_entity)
                        new_entity.dxf.layer = "ORIGINAL_ENTITIES"
                except Exception as e:
                    print(f"警告: 无法复制实体 {entity} - {str(e)}")
                    continue         
            
            # 2. 删除由LINE组成的圆形和CIRCLE类型的圆形
            deleted_count = remove_circular_entities(doc)
            print(f"已删除 {deleted_count} 个圆形实体")
            
            # 3. 保存所有创建的多边形（如果有）
            if hasattr(self, 'polygons') and self.polygons:
                for poly_data in self.polygons:
                    polygon_points = [tuple(round(coord,4) for coord in p) for p in poly_data['polygon'].exterior.coords]
                    msp.add_lwpolyline(
                        polygon_points, 
                        close=True,
                        dxfattribs={
                            'color': random.randint(1, 7),
                            'layer': "POLYGONS"
                        }
                    )
            
            doc.saveas(save_path)
            return True
        except Exception as e:
            print(f"保存DXF文件失败: {e}")
            return False

class PointSelector:
    def __init__(self, filename=None, points=None):
        self.point_manager = DxfPointManager(filename, points)
        
        self.radius_filter = None  # 添加半径过滤器
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        title = f"操作指南: 框选点→V凹包/B保存/D删除上次/C删除点/R设置半径/ -- O打开新文件"
        if filename:
            title = f"文件: {os.path.basename(filename)} - " + title
        self.ax.set_title(title)
        
        self.legend_items = []
        
        # 初始绘制
        self.scatter = self.ax.scatter(
            self.point_manager.available_points[:, 0],
            self.point_manager.available_points[:, 1],
            c='blue', alpha=0.7, s=1
        )
        self.legend_items.append("可用点")
        
        # 绘制已删除的点（如果有）
        if self.point_manager.deleted_indices:
            deleted_points = self.point_manager.original_points[list(self.point_manager.deleted_indices)]
            self.ax.scatter(
                deleted_points[:, 0],
                deleted_points[:, 1],
                c='red', alpha=0.3, s=1, label="已删除点"
            )
            self.legend_items.append("已删除点")
        
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1, 3],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        self.current_polygon_plot = None
        self.current_polygon = None
        self.saved_polygon_plots = []
        
        self.update_legend()
        plt.connect('key_press_event', self.on_key)
        plt.show()
    
    def update_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        if len(labels) > 5:
            handles = handles[-5:]
            labels = labels[-5:]
        self.ax.legend(handles, labels, loc='upper right')
        self.fig.canvas.draw()
    
    def update_scatter(self):
        """更新散点图显示"""
        # 清除所有点
        for coll in self.ax.collections:
            coll.remove()
        
        # 绘制可用点
        available_points = self.point_manager.available_points
        if len(available_points) > 0:
            self.scatter = self.ax.scatter(
                available_points[:, 0],
                available_points[:, 1],
                c='blue', alpha=0.7, s=1, label="可用点"
            )
        
        # 绘制已用点
        used_points = self.point_manager.get_used_points()
        if len(used_points) > 0:
            self.ax.scatter(
                used_points[:, 0],
                used_points[:, 1],
                c='gray', alpha=0.3, s=1, label="已用点"
            )
        
        # 绘制已删除的点
        if self.point_manager.deleted_indices:
            deleted_points = self.point_manager.original_points[list(self.point_manager.deleted_indices)]
            self.ax.scatter(
                deleted_points[:, 0],
                deleted_points[:, 1],
                c='red', alpha=0.3, s=1, label="已删除点"
            )
        
        self.update_legend()

    def set_radius_filter(self):
        """设置半径过滤器，显示可选半径值"""
        # 获取当前可用点的半径值
        active_indices = self.point_manager.get_active_indices()
        unique_radii = np.unique(np.round(self.point_manager.radii[active_indices], 4))
        
        print("\n当前半径过滤器:", self.radius_filter)
        print("可用半径值:", ', '.join([f"{r:.4f}".rstrip('0').rstrip('.') if '.' in f"{r:.4f}" else f"{r:.4f}" for r in unique_radii]))
        print("输入半径范围(格式: min,max 或 单个值)，或留空后按回车取消过滤")

        input_str = input("请输入半径范围: ").strip()
        
        if not input_str:
            self.radius_filter = None
            print("已取消半径过滤")
            return
        
        try:
            if ',' in input_str:
                min_r, max_r = map(float, input_str.split(','))
                self.radius_filter = (min_r, max_r)
                print(f"已设置半径范围: {min_r} ≤ r ≤ {max_r}")
            else:
                r = float(input_str)
                self.radius_filter = (r, r)
                print(f"已设置半径: r = {r}")
        except ValueError:
            print("输入格式错误，请使用 min,max 或 单个数值")
            self.radius_filter = None

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        
        available_points = self.point_manager.available_points
        active_indices = self.point_manager.get_active_indices()
        radii = self.point_manager.radii  # 获取所有点的半径
        
        # 创建位置掩码
        pos_mask = (
            (available_points[:, 0] >= x_min) & 
            (available_points[:, 0] <= x_max) & 
            (available_points[:, 1] >= y_min) & 
            (available_points[:, 1] <= y_max)
        )
        
        # 创建半径掩码
        if self.radius_filter is not None:
            min_r, max_r = self.radius_filter
            radius_mask = (radii[active_indices] >= min_r) & (radii[active_indices] <= max_r)
            combined_mask = pos_mask & radius_mask
        else:
            combined_mask = pos_mask
        
        self.selected_indices = [active_indices[i] for i in np.where(combined_mask)[0]]
        
        # 更新点的颜色
        colors = ['blue'] * len(available_points)
        for i in np.where(combined_mask)[0]:
            colors[i] = 'red'
        
        self.scatter.set_color(colors)
        self.fig.canvas.draw()

    def create_rectangle_from_line(self, points):
        """从共线点创建矩形"""
        line = LineString(points)
        selected_radii = self.point_manager.radii[self.selected_indices]
        avg_radius = np.mean(selected_radii)
        
        # 计算矩形边界
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        # 确定是水平线还是垂直线
        if np.allclose(points[:, 0], points[0, 0]):  # 垂直线
            min_x -= avg_radius
            max_x += avg_radius
            min_y -= avg_radius
            max_y += avg_radius
        elif np.allclose(points[:, 1], points[0, 1]):  # 水平线
            min_x -= avg_radius
            max_x += avg_radius
            min_y -= avg_radius
            max_y += avg_radius
        else:  # 斜线
            buffered = line.buffer(avg_radius, cap_style=2, join_style=2)
            return buffered
        
        return box(min_x, min_y, max_x, max_y)
    
    # def on_key(self, event):
    #     if event.key == 'v' and len(self.selected_indices) >= 2:  # 保留凹包功能
    #         self.create_polygon('concave')
    #     elif event.key == 'c' and self.selected_indices:  # 将删除点功能从F改为C
    #         self.delete_selected_points()
    #     elif event.key == 'b' and self.point_manager.polygons:
    #         self.save_to_dxf()
    #     elif event.key == 'd':
    #         self.delete_last_operation()
    #     elif event.key == 'r':
    #         self.reset_all()
    #     elif event.key == 'o':  # 添加O键打开新文件功能
    #         self.open_new_file()

    def on_key(self, event):
        if event.key == 'v' and len(self.selected_indices) >= 2:
            self.create_polygon('concave')
        elif event.key == 'c' and self.selected_indices:
            self.delete_selected_points()
        elif event.key == 'b' :
            self.save_to_dxf()
        elif event.key == 'd':
            self.delete_last_operation()
        elif event.key == 'r':
            self.set_radius_filter()
        elif event.key == 'o':
            self.open_new_file()

    def open_new_file(self):
        """打开新的DXF文件"""
        filename = select_file()
        if not filename:
            print("未选择文件")
            return

        try:
            # 重置当前状态
            self.reset_all()
            # 加载新文件
            self.point_manager = DxfPointManager(filename)

            # 重新创建选择器，保持非交互状态
            self.selector = RectangleSelector(
                self.ax, self.on_select,
                useblit=True,
                button=[1, 3],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=False
            )
            # 禁用缩放工具
            self.fig.canvas.toolbar.zoom()

            # 更新界面
            self.ax.set_title(f"文件: {os.path.basename(filename)} - 操作指南: 框选点→V凹包/B保存/D删除上次/C删除点/R设置半径/ -- O打开新文件")
            self.update_scatter()
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"加载新文件失败: {e}")
    
    def create_polygon(self, polygon_type):
        self.clear_current_polygon()
        
        selected_points = self.point_manager.original_points[self.selected_indices]
        selected_radii = self.point_manager.radii[self.selected_indices]
        
        # 检查是否只选择了两个点
        if len(selected_points) == 2:
            p1, p2 = selected_points
            avg_radius = np.mean(selected_radii)
            
            # 计算方向向量
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 计算垂直向量
            if abs(dx) > abs(dy):  # 更接近水平线
                perp_dx = 0
                perp_dy = avg_radius
            else:  # 更接近垂直线
                perp_dx = avg_radius
                perp_dy = 0
            
            # 计算矩形尺寸
            length = np.sqrt(dx**2 + dy**2) + avg_radius
            width = avg_radius
            
            # 单位向量
            if length > 0:
                ux = dx / length
                uy = dy / length
            else:
                ux, uy = 1, 0
            
            # 垂直单位向量
            perp_ux = -uy
            perp_uy = ux
            
            # 计算扩展后的端点
            p1_extended = [
                p1[0] - ux * avg_radius/2,
                p1[1] - uy * avg_radius/2
            ]
            p2_extended = [
                p2[0] + ux * avg_radius/2,
                p2[1] + uy * avg_radius/2
            ]
            
            # 计算矩形四个角点
            corner1 = [
                p1_extended[0] - perp_ux * width,
                p1_extended[1] - perp_uy * width
            ]
            corner2 = [
                p1_extended[0] + perp_ux * width,
                p1_extended[1] + perp_uy * width
            ]
            corner3 = [
                p2_extended[0] + perp_ux * width,
                p2_extended[1] + perp_uy * width
            ]
            corner4 = [
                p2_extended[0] - perp_ux * width,
                p2_extended[1] - perp_uy * width
            ]
            
            # 创建多边形
            polygon = Polygon([corner1, corner2, corner3, corner4, corner1])
            if not polygon.is_valid:
                print("警告: 创建的多边形无效!")
            label = '两点矩形'
            color = to_hex((random.random(), random.random(), random.random()))

            # 绘制多边形
            x, y = polygon.exterior.xy
            self.current_polygon_plot, = self.ax.plot(
                x, y, '-', linewidth=1, color=color, label=f'{label}({len(self.point_manager.polygons)+1})'
            )
            
            # 更新点状态
            self.point_manager.use_points(self.selected_indices)
            polygon_data = self.point_manager.add_polygon(polygon, 'two_point_rectangle', color)
            self.point_manager.add_selection(self.selected_indices, polygon_data)
            
            self.update_scatter()
            self.update_legend()
            self.add_current_to_saved()
            return
        
        # 检查是否共线
        if len(selected_points) >= 2:
            line_vec = selected_points[1] - selected_points[0]
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-7:
                line_vec = line_vec / line_len
                perp_vec = np.array([-line_vec[1], line_vec[0]])
                
                distances = []
                for point in selected_points[2:]:
                    vec = point - selected_points[0]
                    dist = abs(np.dot(vec, perp_vec))
                    distances.append(dist)
                
                if all(d < 1e-7 for d in distances):
                    avg_radius = np.mean(selected_radii)
                    polygon = self.create_rectangle_from_line(selected_points)
                    
                    label = '矩形'
                    color = to_hex((random.random(), random.random(), random.random()))
                    # 绘制多边形
                    x, y = polygon.exterior.xy
                    self.current_polygon_plot, = self.ax.plot(
                        x, y, '-', linewidth=1, color=color, label=f'{label}({len(self.point_manager.polygons)+1})'
                    )
                    
                    # 更新点状态
                    self.point_manager.use_points(self.selected_indices)
                    polygon_data = self.point_manager.add_polygon(polygon, 'rectangle', color)
                    self.point_manager.add_selection(self.selected_indices, polygon_data)
                    
                    self.update_scatter()
                    self.update_legend()
                    self.add_current_to_saved()
                    return
        
        # 非共线点处理
        multipoint = MultiPoint(selected_points)
        avg_radius = np.mean(selected_radii)
        
        if polygon_type == 'convex':
            hull = multipoint.convex_hull
            label = '凸包'
        else:
            try:
                hull = concave_hull(multipoint, ratio=0.02)
                label = '凹包'
            except Exception as e:
                hull = multipoint.convex_hull
                label = '凸包(凹包失败)'
        
        if isinstance(hull, Polygon):
            expanded_hull = hull.buffer(avg_radius, join_style=2)
            color = to_hex((random.random(), random.random(), random.random()))

            # 绘制多边形
            x, y = expanded_hull.exterior.xy
            self.current_polygon_plot, = self.ax.plot(
                x, y, '-', linewidth=1, color=color, label=f'{label}({len(self.point_manager.polygons)+1})'
            )
            
            # 更新点状态
            self.point_manager.use_points(self.selected_indices)
            polygon_data = self.point_manager.add_polygon(expanded_hull, polygon_type, color)
            self.point_manager.add_selection(self.selected_indices, polygon_data)
            
            self.update_scatter()
            self.update_legend()
            self.add_current_to_saved()
    
    def delete_selected_points(self):
        """删除选中的点"""
        if not self.selected_indices:
            print("没有选中的点可删除")
            return
        
        self.point_manager.delete_points(self.selected_indices)
        self.selected_indices = []
        self.update_scatter()
    
    def delete_last_operation(self):
        last_selection = self.point_manager.pop_last_selection()
        if last_selection is None:
            print("没有可删除的操作记录")
            return
        
        if last_selection['indices']:
            self.point_manager.restore_points(last_selection['indices'])
        
        if last_selection['polygon_data'] and last_selection['polygon_data'] in self.point_manager.polygons:
            self.point_manager.polygons.remove(last_selection['polygon_data'])
        
        self.update_scatter()
        
        if self.saved_polygon_plots:
            try:
                self.saved_polygon_plots[-1].remove()
            except:
                pass
            self.saved_polygon_plots.pop()
        
        self.update_legend()
    
    def reset_all(self):
        # 恢复所有已用点
        all_used = set()
        for poly in self.point_manager.polygons:
            all_used.update(poly['used_indices'])
        
        self.point_manager.available_indices.update(all_used)
        self.point_manager.used_indices = set()
        self.point_manager.polygons = []
        self.point_manager.selection_history = []
        self.point_manager.deleted_indices = set()  # 恢复所有被删除的点
        
        for plot in self.saved_polygon_plots:
            try:
                plot.remove()
            except:
                pass
        self.saved_polygon_plots = []
        self.clear_current_polygon()
        
        self.update_scatter()
    
    def add_current_to_saved(self):
        if self.current_polygon_plot:
            self.saved_polygon_plots.append(self.current_polygon_plot)
            self.current_polygon_plot = None
            self.update_legend()
    
    def save_to_dxf(self):
        print("正在保存中")
        # 修改条件，即使没有多边形也允许保存
        if not hasattr(self.point_manager, 'polygons') or not hasattr(self.point_manager, 'original_doc'):
            print("没有可保存的内容")
            return
        
        root = tk.Tk()
        root.withdraw()
        default_filename = f"modified_{os.path.basename(self.point_manager.filename)}" if self.point_manager.filename else "output_polygons.dxf"
        save_path = filedialog.asksaveasfilename(
            title="保存DXF文件",
            defaultextension=".dxf",
            initialfile=default_filename,
            filetypes=[("DXF文件", "*.dxf"), ("所有文件", "*.*")]
        )
        
        if not save_path:
            print("保存已取消")
            return
        
        if self.point_manager.save_to_dxf(save_path):
            print(f"保存成功: {save_path}")
    
    def clear_current_polygon(self):
        if self.current_polygon_plot:
            try:
                self.current_polygon_plot.remove()
            except:
                pass
            self.current_polygon_plot = None

def main():
    print("DXF圆点处理工具")
    print("1. 从DXF文件读取\n2. 手动输入坐标")
    
    while True:
        choice = input("选择(1/2): ").strip()
        
        if choice == '1':
            filename = select_file()
            if not filename:
                print("未选择文件，程序退出")
                return
            
            try:
                selector = PointSelector(filename=filename)
                break
            except Exception as e:
                print(f"初始化失败: {e}")
                continue
                
        elif choice == '2':
            print("输入坐标点(x1,y1 x2,y2 ...):")
            
            while True:
                try:
                    input_str = input("坐标点: ").strip()
                    if not input_str:
                        continue
                        
                    points = [tuple(map(float, pair.split(','))) for pair in input_str.split()]
                    
                    if len(points) < 3:
                        print("至少需要3个点")
                        continue
                        
                    selector = PointSelector(points=points)
                    return
                    
                except ValueError:
                    print("格式错误，使用 x1,y1 x2,y2")
                except Exception as e:
                    print(f"错误: {e}")
        else:
            print("请输入1或2")

if __name__ == "__main__":
    main()