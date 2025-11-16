# NANO-OS Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help up down restart logs shell shell-db shell-redis clean test lint format migrate migrate-create migrate-up migrate-down install dev build

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)NANO-OS - Nanomaterials Operating System$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

##@ Docker Commands

up: ## Start all services with docker-compose
	@echo "$(GREEN)Starting NANO-OS services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started! Waiting for health checks...$(NC)"
	@sleep 5
	docker-compose ps

down: ## Stop all services
	@echo "$(YELLOW)Stopping NANO-OS services...$(NC)"
	docker-compose down

restart: ## Restart all services
	@echo "$(YELLOW)Restarting NANO-OS services...$(NC)"
	docker-compose restart

logs: ## Show logs from all services
	docker-compose logs -f

logs-api: ## Show API logs
	docker-compose logs -f api

logs-worker: ## Show worker logs
	docker-compose logs -f worker

logs-db: ## Show database logs
	docker-compose logs -f postgres

ps: ## Show running containers
	docker-compose ps

##@ Database Commands

shell-db: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U orion -d orion_db

shell-redis: ## Open Redis CLI
	docker-compose exec redis redis-cli -a orion_redis_pwd

migrate: migrate-up ## Run database migrations (alias for migrate-up)

migrate-create: ## Create a new migration (usage: make migrate-create MSG="description")
	@if [ -z "$(MSG)" ]; then \
		echo "$(YELLOW)Usage: make migrate-create MSG='description of changes'$(NC)"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(MSG)"

migrate-up: ## Apply all pending migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)Migrations complete!$(NC)"

migrate-down: ## Rollback last migration
	@echo "$(YELLOW)Rolling back last migration...$(NC)"
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

migrate-current: ## Show current migration version
	alembic current

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(YELLOW)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker-compose up -d postgres redis; \
		sleep 5; \
		make migrate-up; \
		echo "$(GREEN)Database reset complete!$(NC)"; \
	fi

##@ Development Commands

install: ## Install Python dependencies
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed!$(NC)"

dev: ## Run API server in development mode
	@echo "$(GREEN)Starting development server...$(NC)"
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

worker: ## Start Celery worker in development mode
	@echo "$(GREEN)Starting Celery worker...$(NC)"
	celery -A src.worker.celery_app worker --loglevel=info

worker-dev: ## Start Celery worker with auto-reload
	@echo "$(GREEN)Starting Celery worker with auto-reload...$(NC)"
	watchmedo auto-restart --directory=./src/worker --pattern=*.py --recursive -- celery -A src.worker.celery_app worker --loglevel=info

flower: ## Start Flower (Celery monitoring)
	@echo "$(GREEN)Starting Flower at http://localhost:5555$(NC)"
	celery -A src.worker.celery_app flower

shell: ## Open Python shell with app context
	python -i -c "from src.api.app import app; from src.api.database import get_db_context; from src.api.models import *; print('App context loaded. Available: app, get_db_context, models')"

##@ Testing Commands

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-api: ## Run API tests only
	pytest tests/test_api.py -v

test-worker: ## Run worker tests only
	pytest tests/test_worker.py -v

test-structures: ## Run structure parser tests
	pytest tests/test_structures.py -v --run-pymatgen

test-fast: ## Run tests (skip slow tests)
	pytest tests/ -v -m "not slow"

test-watch: ## Run tests in watch mode
	pytest-watch tests/ -v

##@ Code Quality Commands

lint: ## Run all linters
	@echo "$(GREEN)Running linters...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run type checking with mypy
	mypy src/

security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r src/
	safety check

##@ Build Commands

build: ## Build Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker-compose build

build-nocache: ## Build Docker images without cache
	@echo "$(GREEN)Building Docker images (no cache)...$(NC)"
	docker-compose build --no-cache

push: ## Push Docker images to registry
	@echo "$(YELLOW)Pushing Docker images...$(NC)"
	docker-compose push

##@ Cleanup Commands

clean: ## Clean up generated files
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-all: clean down ## Clean everything including Docker volumes
	@echo "$(YELLOW)Removing Docker volumes...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)Complete cleanup done!$(NC)"

##@ Frontend Commands

frontend-install: ## Install frontend dependencies
	@echo "$(GREEN)Installing frontend dependencies...$(NC)"
	cd frontend && npm install

frontend-dev: ## Run frontend development server
	@echo "$(GREEN)Starting frontend dev server at http://localhost:3000$(NC)"
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	@echo "$(GREEN)Building frontend...$(NC)"
	cd frontend && npm run build

frontend-lint: ## Lint frontend code
	cd frontend && npm run lint

frontend-test: ## Test frontend
	cd frontend && npm test

##@ Production Commands

prod-up: ## Start production environment
	@echo "$(GREEN)Starting production environment...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-logs: ## Show production logs
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

prod-down: ## Stop production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

##@ Utility Commands

env: ## Create .env file from .env.example
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).env file created from .env.example$(NC)"; \
		echo "$(YELLOW)Please update .env with your configuration$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

status: ## Show service status
	@echo "$(BLUE)NANO-OS Service Status$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Containers:$(NC)"
	@docker-compose ps || echo "$(YELLOW)Docker not running$(NC)"
	@echo ""
	@echo "$(GREEN)Database Status:$(NC)"
	@alembic current 2>/dev/null || echo "$(YELLOW)Alembic not initialized$(NC)"
	@echo ""
	@echo "$(GREEN)API Health:$(NC)"
	@curl -s http://localhost:8000/health 2>/dev/null | python -m json.tool || echo "$(YELLOW)API not responding$(NC)"

health: ## Check health of all services
	@echo "$(GREEN)Checking service health...$(NC)"
	@echo ""
	@echo "API: " && curl -sf http://localhost:8000/health > /dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"
	@echo "Database: " && docker-compose exec -T postgres pg_isready -U orion -d orion_db > /dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"
	@echo "Redis: " && docker-compose exec -T redis redis-cli -a orion_redis_pwd ping > /dev/null 2>&1 && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(YELLOW)✗ Unhealthy$(NC)"

seed: ## Seed database with sample data
	@echo "$(GREEN)Seeding database with sample data...$(NC)"
	python scripts/seed_data.py

backup: ## Backup database
	@echo "$(GREEN)Creating database backup...$(NC)"
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U orion orion_db > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup created in backups/$(NC)"

restore: ## Restore database from backup (usage: make restore FILE=backups/backup.sql)
	@if [ -z "$(FILE)" ]; then \
		echo "$(YELLOW)Usage: make restore FILE=backups/backup.sql$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restoring database from $(FILE)...$(NC)"
	docker-compose exec -T postgres psql -U orion -d orion_db < $(FILE)
	@echo "$(GREEN)Restore complete!$(NC)"

##@ Documentation

docs: ## Generate API documentation
	@echo "$(GREEN)Generating API documentation...$(NC)"
	@echo "API docs available at http://localhost:8000/docs"
	@echo "ReDoc available at http://localhost:8000/redoc"

tree: ## Show project structure
	@echo "$(BLUE)NANO-OS Project Structure:$(NC)"
	tree -L 3 -I '__pycache__|*.pyc|node_modules|.git' .

info: ## Show project information
	@echo "$(BLUE)╔════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║  NANO-OS - Nanomaterials Operating System             ║$(NC)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Services:$(NC)"
	@echo "  • API: http://localhost:8000"
	@echo "  • API Docs: http://localhost:8000/docs"
	@echo "  • Frontend: http://localhost:3000"
	@echo "  • Flower (Celery): http://localhost:5555"
	@echo "  • Neo4j Browser: http://localhost:7474"
	@echo "  • MinIO Console: http://localhost:9001"
	@echo "  • Grafana: http://localhost:3001"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  1. make env          # Create .env file"
	@echo "  2. make up           # Start services"
	@echo "  3. make migrate      # Run migrations"
	@echo "  4. make dev          # Start dev server"
	@echo ""
	@echo "$(GREEN)Common Commands:$(NC)"
	@echo "  • make help          # Show all commands"
	@echo "  • make test          # Run tests"
	@echo "  • make logs          # View logs"
	@echo "  • make status        # Check status"
	@echo ""
