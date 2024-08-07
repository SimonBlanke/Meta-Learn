install:
	python -m pip install -r ./requirements/requirements.txt

install-test:
	python -m pip install -r ./requirements/requirements-test.txt

dev-install:
	pip install -e .

reinstall:
	pip uninstall -y meta-learn
	rm -fr build dist meta-learn.egg-info
	python -m build
	pip install dist/*.whl

tox-test:
	tox -- -x -p no:warnings -rfEX tests/ \

py-test:
	python -m pytest -x -p no:warnings tests/; \

test:  py-test tox-test

requirement:
	cd requirements/; \
		pip-compile requirements.in;\
		pip-compile requirements-test.in