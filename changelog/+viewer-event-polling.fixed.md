Avoid a blocking event-poll wait in the OpenGL viewer's manual frame loop, which can add several milliseconds per frame on Windows despite a nominal one-millisecond timeout.
