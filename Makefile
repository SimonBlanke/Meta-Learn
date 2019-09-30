reinstall:
	pip uninstall -y meta-learn
	rm -fr build dist meta-learn.egg-info
	python setup.py bdist_wheel
	pip install dist/*
