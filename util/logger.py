class Logger:
    def __init__(self):
        self.logger_id = "No logger ID"
    
    def set_logger_id(self, logger_id):
        self.logger_id = logger_id

    def log(self, *args):
        print(f"[{self.logger_id}]", *args)
    
    def error(self, *args):
        print(f"[ERROR][{self.logger_id}]", *args)

logger = Logger()