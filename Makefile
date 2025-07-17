hello:
	@echo 'It works!'

install_requirements:
	@pip install -r requirements.txt

run_api:
	uvicorn fast_api.api:app --reload --port 8000
