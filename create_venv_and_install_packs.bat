python -m venv venv
call .\venv\Scripts\activate
pip install --no-index --ignore-installed --find-links=./packagesdir -r requirements.txt
pause
