from jinja2 import Template
import shutil

input_template_path="testcodes/test.html"
temp_html_path="testcodes/test2.html"
text="<img src='"+"murasaki.jpg"+"'>"

image_jinja_data = {"pic_done":text}
with open(temp_html_path, "w", encoding="utf-8") as temp_file:
    with open(input_template_path, "r", encoding="utf-8") as template_file:
        j2_template = Template(template_file.read())
        temp_file.write(j2_template.render(image_jinja_data))

shutil.copy(temp_html_path, input_template_path)