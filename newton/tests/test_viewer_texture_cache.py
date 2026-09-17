# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check file texture reuse without requiring an OpenGL context."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from newton._src.viewer.gl.opengl import MeshGL, RendererGL


class TestViewerTextureCache(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "texture.png"
        self.path.write_bytes(b"initial")
        self.mesh = MeshGL.__new__(MeshGL)
        self.mesh.texture_id = None
        self.mesh._texture_file_key = None
        self.gl = mock.Mock()
        self.pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        patches = (
            mock.patch.object(RendererGL, "gl", self.gl),
            mock.patch("newton._src.utils.texture.load_texture", return_value=self.pixels),
            mock.patch("newton._src.viewer.gl.opengl._upload_texture_from_file", return_value=17),
        )
        mocks = []
        for patch in patches:
            mocks.append(patch.start())
            self.addCleanup(patch.stop)
        _, self.load, self.upload = mocks

    def test_reuse_and_file_change(self):
        """Reuse unchanged files but reload edits, replacements and removed textures."""
        self.mesh.update_texture(self.path)
        self.mesh.update_texture(str(self.path))
        self.assertEqual(self.load.call_count, 1)
        self.assertEqual(self.upload.call_count, 1)
        self.gl.glDeleteTextures.assert_not_called()
        self.path.write_bytes(b"different image")
        self.mesh.update_texture(self.path)
        self.assertEqual(self.load.call_count, 2)
        self.gl.glDeleteTextures.assert_called_once_with(1, 17)
        self.mesh.update_texture(None)
        self.assertIsNone(self.mesh.texture_id)
        self.mesh.update_texture(self.path)
        self.assertEqual(self.load.call_count, 3)
        self.path.unlink()
        self.load.return_value = None
        self.mesh.update_texture(self.path)
        self.assertIsNone(self.mesh.texture_id)

    def test_mutable_arrays_are_refreshed(self):
        """Continue uploading arrays that may have been modified in place."""
        self.mesh.update_texture(self.pixels)
        self.pixels[0, 0] = 255
        self.mesh.update_texture(self.pixels)
        self.assertEqual(self.upload.call_count, 2)

    def test_failed_upload_is_retried(self):
        """Do not cache a file until its GL upload has succeeded."""
        self.upload.side_effect = [None, 17]
        self.mesh.update_texture(self.path)
        self.mesh.update_texture(self.path)
        self.assertEqual(self.upload.call_count, 2)
        self.assertEqual(self.mesh.texture_id, 17)


if __name__ == "__main__":
    unittest.main()
