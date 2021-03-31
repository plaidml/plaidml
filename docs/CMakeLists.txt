add_custom_target(docs
  COMMAND ${PYTHON_EXECUTABLE} docs/docs.py
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
