import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kirara_ai.im.message import ImageMessage, VoiceMessage
from kirara_ai.media import MediaManager, MediaType


class TestMediaManager(unittest.TestCase):
    """测试媒体管理器"""

    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.media_dir = os.path.join(self.temp_dir, "media")
        
        # 创建各种格式的测试文件
        self.format_files = {}
        
        # 图片格式
        self.format_files["jpeg"] = os.path.join(self.temp_dir, "test.jpg")
        with open(self.format_files["jpeg"], "wb") as f:
            f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C")
            
        self.format_files["png"] = os.path.join(self.temp_dir, "test.png")
        with open(self.format_files["png"], "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89")
            
        self.format_files["gif"] = os.path.join(self.temp_dir, "test.gif")
        with open(self.format_files["gif"], "wb") as f:
            f.write(b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")
            
        self.format_files["webp"] = os.path.join(self.temp_dir, "test.webp")
        with open(self.format_files["webp"], "wb") as f:
            f.write(b"RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x00\x10\x00\x00\x00\x10\x00\x00\x00\x01\x00\x02\x00\x02\x00\x34\x25\xa4\x00\x03p\x00\xfe\xfb\xfd\x50\x00")
            
        # 音频格式
        self.format_files["mp3"] = os.path.join(self.temp_dir, "test.mp3")
        with open(self.format_files["mp3"], "wb") as f:
            f.write(b"\xFF\xFB\x90\x64\x00\x00\x00\x00")
            
        self.format_files["wav"] = os.path.join(self.temp_dir, "test.wav")
        with open(self.format_files["wav"], "wb") as f:
            f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x11+\x00\x00\x11+\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00")
            
        # 视频格式
        self.format_files["mp4"] = os.path.join(self.temp_dir, "test.mp4")
        with open(self.format_files["mp4"], "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\x00moov")
            
        self.format_files["avi"] = os.path.join(self.temp_dir, "test.avi")
        with open(self.format_files["avi"], "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00AVI LIST\x00\x00\x00\x00hdrlavih\x00\x00\x00\x00")
            
        # 文档格式
        self.format_files["pdf"] = os.path.join(self.temp_dir, "test.pdf")
        with open(self.format_files["pdf"], "wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 3 3]/Parent 2 0 R/Resources<<>>>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000015 00000 n\n0000000060 00000 n\n0000000111 00000 n\ntrailer\n<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF\n")
            
        self.format_files["txt"] = os.path.join(self.temp_dir, "test.txt")
        with open(self.format_files["txt"], "wb") as f:
            f.write(b"This is a test text file.")
        
        # 使用已创建的文件作为测试文件
        self.test_image_path = self.format_files["jpeg"]
        self.test_audio_path = self.format_files["mp3"]
        
        # 创建媒体管理器
        self.media_manager = MediaManager(media_dir=self.media_dir)

    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)

    def test_register_from_path(self):
        """测试从文件路径注册媒体"""
        media_id = self.media_manager.register_from_path(
            self.test_image_path,
            source="test",
            description="测试图片",
            tags=["test", "image"],
            reference_id="test_ref"
        )

        # 验证媒体ID是否有效
        self.assertIsNotNone(media_id)
        
        # 验证元数据是否正确
        metadata = self.media_manager.get_metadata(media_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.media_type, MediaType.IMAGE)
        self.assertEqual(metadata.source, "test")
        self.assertEqual(metadata.description, "测试图片")
        self.assertEqual(metadata.tags, ["test", "image"])
        self.assertEqual(metadata.references, {"test_ref"})
        
        # 验证文件是否存在
        file_path = asyncio.run(self.media_manager.get_file_path(media_id))
        self.assertIsNotNone(file_path)
        self.assertTrue(file_path.exists())

    def test_register_from_data(self):
        """测试从二进制数据注册媒体"""
        with open(self.test_image_path, "rb") as f:
            data = f.read()
            
        media_id = self.media_manager.register_from_data(
            data,
            format="jpeg",
            source="test_data",
            description="测试数据图片",
            tags=["test", "data"],
            reference_id="test_data_ref"
        )
        
        # 验证媒体ID是否有效
        self.assertIsNotNone(media_id)
        
        # 验证元数据是否正确
        metadata = self.media_manager.get_metadata(media_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.media_type, MediaType.IMAGE)
        self.assertEqual(metadata.source, "test_data")
        self.assertEqual(metadata.description, "测试数据图片")
        self.assertEqual(metadata.tags, ["test", "data"])
        self.assertEqual(metadata.references, {"test_data_ref"})

    def test_register_from_url(self):
        """测试从URL注册媒体"""
        # 使用本地文件URL作为测试
        file_url = f"file://{Path(self.test_image_path).absolute()}"
        
        media_id = self.media_manager.register_from_url(
            file_url,
            source="test_url",
            description="测试URL图片",
            tags=["test", "url"],
            reference_id="test_url_ref"
        )
        
        # 验证媒体ID是否有效
        self.assertIsNotNone(media_id)
        
        # 验证元数据是否正确
        metadata = self.media_manager.get_metadata(media_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.url, file_url)
        self.assertEqual(metadata.source, "test_url")
        self.assertEqual(metadata.description, "测试URL图片")
        self.assertEqual(metadata.tags, ["test", "url"])
        self.assertEqual(metadata.references, {"test_url_ref"})
        
        # 获取数据（这会触发下载）
        data = asyncio.run(self.media_manager.get_data(media_id))
        self.assertIsNotNone(data)
        
        # 再次检查元数据，应该有更多信息
        metadata = self.media_manager.get_metadata(media_id)
        self.assertIsNotNone(metadata.media_type)
        self.assertIsNotNone(metadata.format)

    def test_format_detection(self):
        """测试不同格式文件的类型检测"""
        # 图片格式测试
        for format_name in ["jpeg", "png", "gif", "webp"]:
            media_id = self.media_manager.register_from_path(
                self.format_files[format_name],
                reference_id=f"test_{format_name}"
            )
            metadata = self.media_manager.get_metadata(media_id)
            self.assertEqual(metadata.media_type, MediaType.IMAGE, f"格式 {format_name} 应该被识别为图片")
            self.assertEqual(metadata.format.lower(), format_name.lower(), f"格式 {format_name} 未被正确识别")
        
        # 音频格式测试
        for format_name in ["mp3", "wav"]:
            media_id = self.media_manager.register_from_path(
                self.format_files[format_name],
                reference_id=f"test_{format_name}"
            )
            metadata = self.media_manager.get_metadata(media_id)
            self.assertEqual(metadata.media_type, MediaType.VOICE, f"格式 {format_name} 应该被识别为音频")
            self.assertEqual(metadata.format.lower(), format_name.lower(), f"格式 {format_name} 未被正确识别")
        
        # 视频格式测试
        for format_name in ["mp4", "avi"]:
            media_id = self.media_manager.register_from_path(
                self.format_files[format_name],
                reference_id=f"test_{format_name}"
            )
            metadata = self.media_manager.get_metadata(media_id)
            self.assertEqual(metadata.media_type, MediaType.VIDEO, f"格式 {format_name} 应该被识别为视频")
            self.assertEqual(metadata.format.lower(), format_name.lower() if format_name != "avi" else "x-msvideo", f"格式 {format_name} 未被正确识别")
        
        # 文档格式测试
        for format_name in ["pdf", "txt"]:
            media_id = self.media_manager.register_from_path(
                self.format_files[format_name],
                reference_id=f"test_{format_name}"
            )
            metadata = self.media_manager.get_metadata(media_id)
            self.assertEqual(metadata.media_type, MediaType.FILE, f"格式 {format_name} 应该被识别为文件")
            expected_format = format_name
            if format_name == "txt":
                expected_format = "plain"
            self.assertTrue(metadata.format.lower().endswith(expected_format.lower()), f"格式 {format_name} 未被正确识别，实际为 {metadata.format}")

    def test_lazy_loading(self):
        """测试懒加载策略"""
        # 注册一个只有URL的媒体
        file_url = f"file://{Path(self.test_image_path).absolute()}"
        url_media_id = self.media_manager.register_from_url(
            file_url,
            reference_id="url_ref"
        )
        
        # 初始元数据应该只有URL
        metadata = self.media_manager.get_metadata(url_media_id)
        self.assertEqual(metadata.url, file_url)
        self.assertIsNone(metadata.media_type)
        self.assertIsNone(metadata.format)
        
        # 获取数据（触发下载和类型检测）
        data = asyncio.run(self.media_manager.get_data(url_media_id))
        self.assertIsNotNone(data)
        
        # 再次检查元数据，应该有更多信息
        metadata = self.media_manager.get_metadata(url_media_id)
        self.assertIsNotNone(metadata.media_type)
        self.assertIsNotNone(metadata.format)
        
        # 获取文件路径（文件应该已经存在）
        file_path = asyncio.run(self.media_manager.get_file_path(url_media_id))
        self.assertIsNotNone(file_path)
        self.assertTrue(file_path.exists())

    def test_reference_management(self):
        """测试引用管理"""
        # 注册媒体
        media_id = self.media_manager.register_from_path(
            self.test_image_path,
            reference_id="ref1"
        )
        
        # 添加引用
        self.media_manager.add_reference(media_id, "ref2")
        
        # 验证引用是否添加成功
        metadata = self.media_manager.get_metadata(media_id)
        self.assertEqual(metadata.references, {"ref1", "ref2"})
        
        # 移除引用
        self.media_manager.remove_reference(media_id, "ref1")
        
        # 验证引用是否移除成功
        metadata = self.media_manager.get_metadata(media_id)
        self.assertEqual(metadata.references, {"ref2"})
        
        # 移除最后一个引用，媒体应该被删除
        self.media_manager.remove_reference(media_id, "ref2")
        
        # 验证媒体是否被删除
        self.assertIsNone(self.media_manager.get_metadata(media_id))

    def test_search(self):
        """测试搜索功能"""
        # 注册多个媒体
        media_id1 = self.media_manager.register_from_path(
            self.test_image_path,
            source="source1",
            description="description with keyword1",
            tags=["tag1", "common"],
            reference_id="ref1"
        )
        
        media_id2 = self.media_manager.register_from_path(
            self.test_audio_path,
            source="source2",
            description="description with keyword2",
            tags=["tag2", "common"],
            reference_id="ref2"
        )
        
        # 根据标签搜索
        results = self.media_manager.search_by_tags(["tag1"])
        self.assertEqual(results, [media_id1])
        
        results = self.media_manager.search_by_tags(["common"])
        self.assertEqual(set(results), {media_id1, media_id2})
        
        # 根据描述搜索
        results = self.media_manager.search_by_description("keyword1")
        self.assertEqual(results, [media_id1])
        
        # 根据来源搜索
        results = self.media_manager.search_by_source("source2")
        self.assertEqual(results, [media_id2])
        
        # 根据类型搜索
        results = self.media_manager.search_by_type(MediaType.IMAGE)
        self.assertEqual(results, [media_id1])
        
        results = self.media_manager.search_by_type(MediaType.VOICE)
        self.assertEqual(results, [media_id2])

    def test_media_message(self):
        """测试MediaMessage类"""
        # 创建只有URL的媒体消息
        file_url = f"file://{Path(self.test_image_path).absolute()}"
        url_message = ImageMessage(url=file_url, reference_id="url_message_ref", media_manager=self.media_manager)
        
        # 验证媒体ID
        self.assertIsNotNone(url_message.media_id)
        
        # 获取URL（应该直接返回原始URL）
        url = asyncio.run(url_message.get_url())
        self.assertEqual(url, file_url)
        
        # 获取路径（应该触发下载）
        path = asyncio.run(url_message.get_path())
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())
        
        # 创建只有路径的媒体消息
        path_message = ImageMessage(path=self.test_image_path, reference_id="path_message_ref", media_manager=self.media_manager)
        
        # 验证媒体ID
        self.assertIsNotNone(path_message.media_id)
        
        # 获取路径（应该直接返回原始路径或复制后的路径）
        path = asyncio.run(path_message.get_path())
        self.assertIsNotNone(path)
        
        # 获取URL（应该生成URL）
        url = asyncio.run(path_message.get_url())
        self.assertIsNotNone(url)
        
        # 创建只有数据的媒体消息
        with open(self.test_image_path, "rb") as f:
            data = f.read()
        
        data_message = ImageMessage(data=data, format="jpeg", reference_id="data_message_ref", media_manager=self.media_manager)
        
        # 验证媒体ID
        self.assertIsNotNone(data_message.media_id)
        
        # 获取数据（应该直接返回原始数据）
        message_data = asyncio.run(data_message.get_data())
        self.assertEqual(message_data, data)
        
        # 获取路径（应该生成文件）
        path = asyncio.run(data_message.get_path())
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).exists())

    def test_media_message_with_different_formats(self):
        """测试不同格式的媒体消息创建"""
        # 测试不同格式的图片
        for format_name in ["jpeg", "png", "gif", "webp"]:
            message = ImageMessage(path=self.format_files[format_name], reference_id=f"message_{format_name}_ref", media_manager=self.media_manager)
            self.assertIsNotNone(message.media_id)
            self.assertEqual(message.resource_type, "image")
            
            # 获取元数据
            metadata = self.media_manager.get_metadata(message.media_id)
            self.assertEqual(metadata.media_type, MediaType.IMAGE)
        
        # 测试不同格式的音频
        for format_name in ["mp3", "wav"]:
            message = VoiceMessage(path=self.format_files[format_name], reference_id=f"message_{format_name}_ref", media_manager=self.media_manager)
            self.assertIsNotNone(message.media_id)
            self.assertEqual(message.resource_type, "audio")
            
            # 获取元数据
            metadata = self.media_manager.get_metadata(message.media_id)
            self.assertEqual(metadata.media_type, MediaType.VOICE)
            
if __name__ == "__main__":
    unittest.main() 