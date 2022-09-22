from controller import controller

app = controller.get_flask_app()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)