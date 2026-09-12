FROM python:3.12-slim
WORKDIR /app
ENV SDL_VIDEODRIVER=dummy MPLBACKEND=Agg PYGAME_HIDE_SUPPORT_PROMPT=1
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "scripts/run_demo.py", "--headless", "--steps", "100", "--snapshot", "results/demo.png"]
