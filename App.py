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
temporary_html_path=r"./temporary_html/temporary.html"


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
    #受け取ったファイルのうち最初のやつだけ抽出し、ほんとにファイルが送られてきたか確認
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    #ファイルが存在するなら、saveメソッドでPics_Readyに保存　WerkZeugのMultiDictの文法注意
    if file:
        filelist=request.files.getlist('file')
        for f in filelist:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(filepath)

        #Pics_Ready内に保存されたすべての画像に対しHLSColorMappingを実行
        imagepath_list = sorted(Path(imgfolder).glob("*"))
        print(imagepath_list)
        for i in imagepath_list:
            imgpath=str(i)
            imgname=imgpath.split("\\")[-1]
            done_imgpath=str(Path(imgfolder_done +"/"+ imgname))
            output_html_path=str(Path(graph_html_folder+r"/{}.html")).format(imgname.split(".")[0])
            #処理する画像は先にPics_Doneへ　HLSColorMapping側でhtml_outputやるときにDone_pics内の画像参照させてHTML作るから
            shutil.move(imgpath, done_imgpath)
            mapping(done_imgpath, input_template_path, output_html_path)
        
        return f'すべてのファイルがアップロードされました: {filepath} 解析結果を保存しました: {graph_html_folder}'
    


if __name__ == '__main__':
    app.run(debug=True)