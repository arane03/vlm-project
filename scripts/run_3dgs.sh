read -rp "Select scene: " SCENE
read -rp "Enter number of iterations: " ITERATIONS

uv run gs-pipeline train --data dataset/${SCENE}/prepared --output models/${SCENE}/model-${ITERATIONS} --iterations ${ITERATIONS} --eval
