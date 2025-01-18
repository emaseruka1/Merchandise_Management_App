from flask import Flask, request,render_template,redirect, session, url_for,flash,send_file,Response
from models import *
import locale
app = Flask(__name__)


df = load_df(path_to_df="E:/WebApp_Projects/IBT_dashboard/Book1.csv")

@app.route("/", methods=['GET','POST'])

def dashboard():

    
    plot_html5 = my_map()

    if request.method=='POST':

        to_store = request.form.get('to_store')
        date_from = request.form.get('date_from')
        date_to = request.form.get('date_to')

        filtered_df = filter_df(df,to_store=to_store,date_from=date_from,date_to=date_to)

        small_df = bubble_df(filtered_df)

        plot_html = plot_bubble_volume(small_df)

        plot_html2 = plot_bubble_cost(small_df)
    

        total_value_moved = small_df['Cost'].sum()
        total_value_moved = f"£ {total_value_moved:,.2f}"
            
        total_volume_moved = small_df['Volume'].sum()
        total_volume_moved = f"{total_volume_moved:,.0f}"

        IBT_count = len(set(filtered_df['Auth Code']))

        

        plot_html4 = line_graph(filtered_df)
    
    else:
        filtered_df = filter_df(df)

        small_df = bubble_df(filtered_df)

        plot_html = plot_bubble_volume(small_df)
        plot_html2 = plot_bubble_cost(small_df)
    

        total_value_moved = small_df['Cost'].sum()
        total_value_moved = f"£ {total_value_moved:,.2f}"
            
        total_volume_moved = small_df['Volume'].sum()
        total_volume_moved = f"{total_volume_moved:,.0f}"

        IBT_count = len(set(filtered_df['Auth Code']))


        plot_html4 = line_graph(df)
        
    

    return render_template("home.html", plot_html=plot_html,plot_html2=plot_html2,small_df=small_df,
                           total_value_moved=total_value_moved,total_volume_moved=total_volume_moved,IBT_count=IBT_count,plot_html4=plot_html4,plot_html5=plot_html5)

@app.route("/network_graph", methods=['GET'])

def net_graph():

    network_graph(df)

    return render_template("network_graph.html")







if __name__=="__main__":
    app.run(host='0.0.0.0',port=5555,debug=True)
    


