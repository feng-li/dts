all: zip

zip:
	git archive --output=dts.zip HEAD
