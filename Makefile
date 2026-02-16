.PHONY: all prep prep_python prep_file histogram correlation points \
standard normal split clean fclean

all: prep

prep: prep_python prep_file
	-@echo "source ./.venv/bin/activate"

prep_file:
	-@rm -rf Test_knight.csv Train_knight.csv
	@wget https://cdn.intra.42.fr/document/document/33135/Test_knight.csv
	@wget https://cdn.intra.42.fr/document/document/33136/Train_knight.csv

prep_python:
	-@rm -rf .venv
	-@chmod +x prep_python.sh
	./prep_python.sh

histogram:
	-@chmod +x ./Histogram.py
	./Histogram.py

correlation:
	-@chmod +x ./Correlation.py
	./Correlation.py

points:
	-@chmod +x ./points.py
	./points.py

standard:
	-@chmod +x ./standardization.py
	./standardization.py

normal:
	-@chmod +x ./Normalization.py
	./Normalization.py

split:
	-@chmod +x ./split.py
	./split.py

clean:
	-rm -rf *.png

fclean: clean
	-@rm -rf Test_knight.csv Train_knight.csv Training_knight.csv Validation_knight.csv
	-rm -rf .venv
