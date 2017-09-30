NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	@for nb in $(NB); do \
		echo "$$(jq --indent 1 ' \
			.metadata = {} \
			| (.cells[] | select(has("outputs")) | .outputs) = [] \
			| (.cells[] | select(has("execution_count")) | .execution_count) = null \
			| .cells[].metadata = {} \
			' $$nb)" > $$nb; \
	done

# May be useful to keep for nbsphinx.
# | .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} \

install:
	pip install --upgrade pip
	pip install --upgrade numpy
	pip install --upgrade -r requirements.txt

readme:
	grip README.md

html:
	grip --export README.md
	jupyter nbconvert $(NB) --to html

.PHONY: run $(NB) clean install readme html
