from flask import Flask
import views

app = Flask(__name__)

app.add_url_rule('/base','base',views.base)
app.add_url_rule('/','home',views.home)
app.add_url_rule('/about','about',views.about)
app.add_url_rule('/home/app','gender',views.gender,methods=['GET','POST'])
if __name__ == "__main__":
    app.run()
