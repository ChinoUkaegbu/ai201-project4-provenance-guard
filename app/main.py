from flask import Flask
from app.extensions import limiter


def create_app() -> Flask:
    app = Flask(__name__)

    limiter.init_app(app)

    from app.routes.submit import submit_bp

    # from app.routes.appeals import appeals_bp
    # from app.routes.log import log_bp

    app.register_blueprint(submit_bp)
    # app.register_blueprint(appeals_bp)
    # app.register_blueprint(log_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
