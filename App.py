#requestオブジェクトでユーザからHTMLにPOSTされたデータを受け取る
#render_templateはtemplateフォルダ内のhtmlをブラウザに表示する
#redirectは指定されたURLのページにユーザを飛ばす
from flask import Flask, render_template, request, redirect, send_from_directory
import os
import shutil
from pathlib import Path
from HLSColorMapping import mapping

app = Flask("HLSColorMapping")

#Pics_Readyというフォルダにアップロードされた画像を保存
UPLOAD_FOLDER = 'Pics_Ready'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#保存先のディレクトリが存在しない場合は作成
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

imgfolder=r"./Pics_Ready"
imgfolder_done=r"./Pics_Done"
input_template_path = r"./templates/result_template.html"
graph_html_folder = r"./Graph_html"

@app.route('/')
def home():
    return render_template('index.html')

#request.filesでHTMLにポストされたファイルたちを辞書として受け取る
#request.urlは/uploadページのことを指す
@app.route('/upload', methods=['POST'])
def upload_file():
    #ファイルがアップロードされないまたはファイル名がない場合は/uploadページにリダイレクトしてエラー吐く
    if 'file' not in request.files:
        return redirect(request.url)
    #今回はまだ1ファイル対応なので受け取ったファイルのうち最初のやつだけ抽出
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    #ファイルが存在するなら、saveメソッドでPics_Readyに保存
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        imagepath_list = list(sorted(str(Path(imgfolder).glob("*"))))
        print(imagepath_list)
        for imgpath in imagepath_list:
            output_html_path=str(Path(graph_html_folder+r"/{}.html")).format(imgpath.split("/")[-1].split(".")[0])
            mapping(imgpath, input_template_path, output_html_path)
            shutil.move(imgpath, Path(imgfolder_done +"/"+ imgpath.split("/")[-1]))
            return send_from_directory('other_templates', output_html_path)


if __name__ == '__main__':
    app.run(debug=True)