TARGET := 'code'
MAKEFILE_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))
PARENT_DIR := $(shell dirname ${MAKEFILE_DIR})
EXCLUDE := .venv

empty :=
space := $(empty) $(empty)

.PHONY: run-formatter
run-formatter:
	$(eval EXCLUDE := $(addprefix -name,$(EXCLUDE)))
	$(eval EXCLUDE := $(subst $(space),$(space)-o$(space),$(strip $(EXCLUDE))))
	$(eval EXCLUDE := $(subst -name,-name$(space),$(strip $(EXCLUDE))))
	$(eval files := $(shell find $(TARGET) -type d \( $(EXCLUDE) \) -prune -false -o -name '*.py'))
	isort --sl --settings-path ${MAKEFILE_DIR}/.isort.cfg ${files}
	autoflake -i --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables ${files}
	black ${files}
	isort -m 3 --ca --settings-path ${MAKEFILE_DIR}/.isort.cfg ${files}

.PHONY: admin
admin:
	sudo chmod 777 *
	sudo chmod 777 code/*.py
	sudo chmod 777 code/**/*.py
	sudo chmod 777 code/**/**/*.py
	sudo chmod 777 code/**/**/**/*.yaml

.PHONY: run-cloudbuild-locally
run-cloudbuild-locally:
	cloud-build-local --config=${MAKEFILE_DIR}/ops/cloudbuild.yaml --dryrun=false ${PARENT_DIR}

generate-requirements:
	poetry export -f requirements.txt --output requirements.txt
