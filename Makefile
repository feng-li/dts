all: zip

zip:
	git archive --output=dts.zip HEAD

install:
	pip install --editable .

update-requirements:
	pip install pipreqs -U
	pipreqs ./ --force
