"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st
class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = None

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        if self.apps is None:
            self.apps = []
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self,df,st_state):
        st.sidebar.header('Navigation')
        titles = [app['title'] for app in self.apps]
        app = st.sidebar.selectbox(
            'Select an app:',
            self.apps,
            format_func=lambda app: app['title'],key='title')
        st.sidebar.write('---')
        
        app['function'](df,st_state)
        

    #st.set_page_config(page_title='QSAR Modeling Web App',
    #    layout='wide')
