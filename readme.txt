
dist文件夹下提供了打包好的exe可执行文件，可直接双击运行，不需要搭建环境

若需要调试源代码，参考如下
运行环境搭建说明：
（本脚本基于python 3.9.9 amd64版本开发）
电脑上没有Python，请先运行python-3.9.9-amd64.exe安装python（勾选添加到PATH）
安装python后，双击create_venv_and_install_packs.bat创建虚拟环境，并在虚拟环境中安装所需要的包
安装完成后，即可双击enter_venv_and_run_pyscript.bat，会自动进入虚拟环境并运行python脚本
如需打包成exe文件，双击build.bat即可，会自动调用pyinstaller进行打包

开始使用吧！
