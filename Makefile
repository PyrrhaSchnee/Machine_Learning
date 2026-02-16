.PHONY: all prep prep_python prep_file histogram correlation points \
standard normal split clean fclean

all: prep

prep: prep_python prep_file
	-@echo "source ./.venv/bin/activate"

prep_file:
	-@rm -rf Test_knight.csv Train_knight.csv truth.txt predictions.txt
	@wget https://cdn.intra.42.fr/document/document/17545/Train_knight.csv
	@wget https://cdn.intra.42.fr/document/document/17543/Test_knight.csv
	@wget https://cdn.intra.42.fr/document/document/17544/predictions.txt
	@wget https://cdn.intra.42.fr/document/document/17542/truth.txt

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

matrix:
	-@chmod +x ./Confusion_Matrix.py
	./Confusion_Matrix.py

heatmap:
	-@chmod +x ./Heatmap.py
	./Heatmap.py

variance:
	-@chmod +x ./Feature_Selection.py
	./Feature_Selection.py

tree:
	-@chmod +x ./Tree.py
	./Tree.py

# voter:
# 	-@chmod +x ./voter.py
# 	./voter.py

clean:
	-rm -rf *.png heatmap_corr.csv Tree.txt KNN.txt Validation_knight.csv Training_knight.csv

fclean: clean
	-@rm -rf .venv Test_knight.csv Train_knight.csv Training_knight.csv Validation_knight.csv truth.txt predictions.txt
