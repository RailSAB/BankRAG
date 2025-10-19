FROM langflowai/langflow:latest


RUN mkdir /app/flows
RUN mkdir /app/langflow-config-dir
WORKDIR /app

COPY flows /app/flows
COPY docker.env /app/.env

ENV PYTHONPATH=/app
ENV LANGFLOW_LOAD_FLOWS_PATH=/app/flows
ENV LANGFLOW_CONFIG_DIR=/app/langflow-config-dir
ENV LANGFLOW_COMPONENTS_PATH=/app/components
ENV LANGFLOW_LOG_ENV=container
RUN pip install langchain_weaviate

EXPOSE 7860
CMD ["langflow", "run", "--env-file","/app/.env","--host", "0.0.0.0", "--port", "7860"]