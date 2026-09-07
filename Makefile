.PHONY: help sync dev add react

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

sync:  ## Install / update dependencies for local LLM development
	uv sync --extra local-llm

dev:   ## Start server with hot-reload
	uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

add:
	## Add dependencies (usage: make add pandas torch)
	uv add $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))

react:
	npm run dev --prefix frontend/rag-app
