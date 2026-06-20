.PHONY: up down logs test lint clean rebuild

# Start all services
up:
	docker-compose up -d --build

# Stop all services
down:
	docker-compose down

# View logs (follow mode)
logs:
	docker-compose logs -f

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Lint code
lint:
	python -m ruff check sentinel/ tests/

# Remove volumes and containers
clean:
	docker-compose down -v --remove-orphans

# Rebuild and restart
rebuild:
	docker-compose down
	docker-compose up -d --build

# View API logs only
logs-api:
	docker-compose logs -f sentinel-api

# View worker logs only
logs-worker:
	docker-compose logs -f sentinel-worker

# Initialize the database schema
init-db:
	python scripts/init_db.py

# Health check
health:
	curl -s http://localhost:8000/api/health | python -m json.tool
