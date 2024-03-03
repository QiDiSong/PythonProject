# 使用Python解码UTF-16编码的字符串
import struct

# 给定的UTF-16编码，忽略了开头的0xff08
encoded_str = b'\xff\x08\x51\x4b\x59\x39\x76\xae'

# 解码UTF-16编码的字符串（假设大端序）
decoded_str = encoded_str.decode('utf-16be')

print(decoded_str)
