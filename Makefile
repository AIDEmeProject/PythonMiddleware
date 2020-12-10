clean:
	rm -f -r build/
	find aideme/ -name "*.so" -type f -delete

clean-c:
	find aideme/ -name "*.c" -type f -delete

clean-all: clean clean-c

.PHONY: build
build: clean
	python setup.py build_ext --inplace

.PHONY: cython-build
cython-build: clean clean-c
	python setup.py build_ext --inplace --use-cython
