.PHONY: help sync dev add

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

sync:  ## Install / update dependencies
	uv sync

dev:   ## Start server with hot-reload
	uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

add:   ## Add a dependency (usage: make add pkg=pandas)
	uv add $(pkg)