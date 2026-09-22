all: zip

zip:
	git archive --output=dts.zip HEAD

install:
	pip install --editable .

# Run in a clean venv containing only this project's dependencies.
update-requirements:
	pip freeze --exclude-editable > requirements.txt
