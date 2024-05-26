from ScreenWriter import ScreenWriter

model = ScreenWriter()

image_file_path = input()
relevant_doc_path = input()
action_prompt = input()

plan = model.predict(image_file_path, relevant_doc_path, action_prompt)

for item in plan:
    print(item)