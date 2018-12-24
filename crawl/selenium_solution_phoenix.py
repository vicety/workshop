from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import json
import re

from selenium_utils import *


# 浏览器配置
option = webdriver.ChromeOptions()
option.add_argument('headless')
# https://stackoverflow.com/questions/46744968/how-to-suppress-console-error-warning-info-messages-when-executing-selenium-pyth
option.add_argument('log-level=1')
driver = webdriver.Chrome(chrome_options=option)
# driver = webdriver.Chrome()

driver.implicitly_wait(2.5)

# 质量限制
max_text_length = 2000
min_text_length = 20
min_img_num = 1
max_img_length = 25
min_comment_num = 3


def is_restriction_satisfied(text_len, img_num):
    return min_text_length <= text_len <= max_text_length and \
           min_img_num <= img_num <= max_img_length


succ_cnt = 0
rec_di = {}
now_url = 'http://news.ifeng.com/listpage/11574/20181102/1/rtlist.shtml'
startday = '20181102'

while True:
    # 整页信息
    print('new page')
    # 某日新闻
    driver.get(now_url)
    nowday = re.search('.*/(.*?)/1/rtlist.shtml', now_url).group(1)
    news_selectors = driver.find_elements_by_css_selector('body div.main div.left div ul li a')
    news_urls = [[''] for _ in range(len(news_selectors))]
    for i, news_selector in enumerate(news_selectors):
        news_urls[i] = news_selector.get_attribute('href')
    nxt_page_url = driver.find_element_by_css_selector('#backDay a').get_attribute('href')
    now_url = nxt_page_url
    for news_url in news_urls:
        try:
            news_id = re.search('.*/(.*?)_0.shtml', news_url).group(1)
            if not news_url:
                continue
            news_item = dict()
            driver.get(news_url)
            context_selector = driver.find_element_by_css_selector('#main_content')
            context = context_selector.text
            text_len = len(context)
            # 排除文章最后的网站logo
            img_selectors = driver.find_elements_by_css_selector('#main_content p img')[:-1]
            img_num = len(img_selectors)
            title = driver.find_element_by_css_selector('#artical_topic').text
            if not is_restriction_satisfied(text_len, img_num):
                print('unsatisfied')
                continue
            # 限制被满足，准备dict
            news_item['title'] = title
            news_item['text'] = context
            news_item['imgs'] = [[''] for _ in range(img_num)]
            for i, selector in enumerate(img_selectors):
                img_url = selector.get_attribute('src')
                news_item['imgs'][i] = img_url
            # 访问评论界面
            # 下滑到底部以加载评论区
            down_y = 0
            for _ in range(20):
                down_y += 800
                js = "var q=document.documentElement.scrollTop={}".format(down_y)
                driver.execute_script(js)
            news_comment_page = driver.find_element_by_css_selector(
                '#js_cmtContainer div.js_checkMoreBlock div a.seeMore').get_attribute('href')
            driver.get(news_comment_page)
            top_comment_selectors = \
                driver.find_elements_by_css_selector('#js_cmtContainer div.js_hotCmtBlock div.mod-commentNew.js_cmtList'
                                                     ' div div p.w-contentTxt')
            # js_cmtContainer > div.js_hotCmtBlock > div.mod-commentNew.js_cmtList > div:nth-child(1) > div > p.w-contentTxt
            top_comment_praise_selectors = \
                driver.find_elements_by_css_selector(
                    '#js_cmtContainer div.js_hotCmtBlock div.mod-commentNew.js_cmtList div div div.w-bottomBar '
                    'span.w-reply a.w-rep-rec.js_recm em')
            # 评论去重
            comment_set = set()
            for i in range(len(top_comment_selectors)):
                if not top_comment_selectors[i].text or not top_comment_praise_selectors[i].text:
                    continue
                comment_set.add((top_comment_selectors[i].text, top_comment_praise_selectors[i].text))
            if len(comment_set) < min_comment_num:
                print('unsatisfied')
                continue
            news_item['top_comments'] = [[''] for _ in range(len(comment_set))]
            news_item['top_comments_parise'] = [[''] for _ in range(len(comment_set))]
            for i, item in enumerate(comment_set):
                news_item['top_comments'][i], news_item['top_comments_parise'][i] = item
            # 打包item
            rec_di[news_id] = news_item

            succ_cnt += 1
            print('*')
            if not succ_cnt % 10:
                print(succ_cnt)
                with open('save/{}_{}.json'.format(startday, nowday), 'w') as f:
                    json.dump(rec_di, f)
        except:
            pass
    driver.get(nxt_page_url)
