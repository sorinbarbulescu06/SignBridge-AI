from ui.app_window import SignLanguageApp

if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
