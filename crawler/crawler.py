import os
import sys
import time
import urllib
import requests
import re
from bs4 import BeautifulSoup
import time


class MakeCrawler:
    def __init__(self):
        self.header = {
            'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 '
                'UBrowser/6.1.2107.204 Safari/537.36'
        }
        self.url = "https://cn.bing.com/images/async?q={0}&first={1}&count={" \
              "2}&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1&dgState=c*9_y" \
              "*2226s2180s2072s2043s2292s2295s2079s2203s2094_i*71_w*198&IG=0D6AD6CBAF43430EA716510A4754C951&SFX={" \
              "3}&iid=images.5599"

    def start(self, crawl_number: int, key_word: str, save_path="./crawler/image/", crawled_number=0):
        def get_image(url, count, path):
            try:
                time.sleep(0.5)
                urllib.request.urlretrieve(url, path + str(count + 1) + '.jpg')
            except Exception as e:
                time.sleep(1)
                print("图像因为未知原因无法保存，正在跳过...")
                return count
            else:
                print("图片+1，已保存 " + str(count + 1) + " 张图")
                return count + 1

        # ----------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------
        # 找到原图并返回URL
        def findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx, count, path, crawl_number):
            soup = BeautifulSoup(html, "lxml")
            link_list = soup.find_all("a", class_="iusc")
            url = []
            for link in link_list:
                result = re.search(rule, str(link))
                # 将字符串"amp;"删除
                url = result.group(0)
                # 组装完整url
                url = url[8:len(url)]
                # 打开高清图片网址
                count = get_image(url, count, path)
                if count >= crawl_number:
                    break
            # 完成一页，继续加载下一页
            return count

        # ----------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------
        # 获取缩略图列表页
        def getStartHtml(url, key, first, loadNum, sfx):
            # 打开页面
            page = urllib.request.Request(url.format(key, first, loadNum, sfx), headers=self.header)
            html = urllib.request.urlopen(page)
            return html

        save_path += key_word + "/"
        # 将关键词转化成URL编码
        key = urllib.parse.quote(key_word)
        # URL中的页码（这里的页码有些抽象，指的是从第几个图片开始）
        first = 1
        # URL中每页的图片数
        loadNum = 35
        # URL中迭代的图片位置（即每页第几个图片）
        sfx = 1
        # 用正则表达式去匹配图片的URL
        rule = re.compile(r"\"murl\":\"http\S[^\"]+")
        # 没有目录就创建目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 开始爬取
        while crawled_number < crawl_number:
            # 获取当前页的html内容
            html = getStartHtml(self.url, key, first, loadNum, sfx)
            # 获取图片的URL并保存图片
            crawled_number = findImgUrlFromHtml(html, rule, self.url, key, first, loadNum, sfx, crawled_number, save_path,
                                                crawl_number)
            # 防止爬取之前的图片
            first = crawled_number + 1

            sfx += 1
        print("爬取成功！已经完成关键词为{0:}的图片爬取{1:}张".format(key_word, crawl_number))