import pyglet.gl


def test_pyglet_gl_imports():
    try:
        # Constants
        assert pyglet.gl.GL_BLEND is not None
        assert pyglet.gl.GL_LINE_LOOP is not None
        assert pyglet.gl.GL_LINE_SMOOTH is not None
        assert pyglet.gl.GL_LINE_SMOOTH_HINT is not None
        assert pyglet.gl.GL_LINE_STIPPLE is not None
        assert pyglet.gl.GL_LINE_STRIP is not None
        assert pyglet.gl.GL_LINES is not None
        assert pyglet.gl.GL_NICEST is not None
        assert pyglet.gl.GL_ONE_MINUS_SRC_ALPHA is not None
        assert pyglet.gl.GL_POINTS is not None
        assert pyglet.gl.GL_POLYGON is not None
        assert pyglet.gl.GL_QUADS is not None
        assert pyglet.gl.GL_SRC_ALPHA is not None
        assert pyglet.gl.GL_TRIANGLES is not None

        # Functions
        assert callable(pyglet.gl.glBegin)
        assert callable(pyglet.gl.glBlendFunc)
        assert callable(pyglet.gl.glClearColor)
        assert callable(pyglet.gl.glColor4f)
        assert callable(pyglet.gl.glDisable)
        assert callable(pyglet.gl.glEnable)
        assert callable(pyglet.gl.glEnd)
        assert callable(pyglet.gl.glHint)
        assert callable(pyglet.gl.glLineStipple)
        assert callable(pyglet.gl.glLineWidth)
        assert callable(pyglet.gl.glPopMatrix)
        assert callable(pyglet.gl.glPushMatrix)
        assert callable(pyglet.gl.glRotatef)
        assert callable(pyglet.gl.glScalef)
        assert callable(pyglet.gl.glTranslatef)
        assert callable(pyglet.gl.gluOrtho2D)
        assert callable(pyglet.gl.glVertex2f)
        assert callable(pyglet.gl.glVertex3f)

        print("All pyglet.gl imports are correct!")
    except AssertionError:
        print("One or more pyglet.gl imports are incorrect!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    test_pyglet_gl_imports()
