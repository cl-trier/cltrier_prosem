
deploy:
	flit publish

test:
	@python3 -m pytest tests/
