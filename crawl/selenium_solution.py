from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import json

from selenium_utils import *

# 浏览器配置
option = webdriver.ChromeOptions()
option.add_argument('headless')
option.add_argument('log-level=1')
driver = webdriver.Chrome(chrome_options=option)
# driver = webdriver.Chrome()
# driver = webdriver.PhantomJS(executable_path=)
driver.implicitly_wait(2.5)

# print('设置浏览器全屏打开')
# driver.maximize_window()
# a = driver.find_element_by_css_selector('body div.gallery img.firstPreload')
# print(a.get_attribute('src'))

succ_cnt = 0
rec_di = {}
now_page = 2292971
start_page = now_page
page_types = ['P', 'O', 'N']

while True:
    succ = False
    # 整页信息
    try:
        for page_type in page_types:
            driver.get("http://news.163.com/photoview/00A{}0001/{}.html".format(page_type, now_page))
            if not element_num(driver, 'div.denominator'):
                continue
            # 成功访问
            rec_di[str(now_page)] = dict()
            rec_di[str(now_page)]['type'] = page_type

            # 页面整体属性
            denominator = int(driver.find_element_by_css_selector('div.denominator').text)
            title = driver.find_element_by_css_selector('div.gallery div.top.cf div.headline h1').text
            desc = driver.find_element_by_css_selector('div.gallery div.main div.sidebar h2 div.viewport div p').text
            rec_di[str(now_page)]['title'] = title
            rec_di[str(now_page)]['desc'] = desc
            rec_di[str(now_page)]['img_desc'] = [[''] for _ in range(denominator)]
            rec_di[str(now_page)]['img_link'] = [[''] for _ in range(denominator)]

            # 标签
            label_selectors = driver.find_elements_by_css_selector('div.gallery div.main div.sidebar div.tag a')
            rec_di[str(now_page)]['labels'] = [[''] for _ in range(len(label_selectors))]
            for i, label_selector in enumerate(label_selectors):
                rec_di[str(now_page)]['labels'][i] = label_selector.text

            # 评论部分
            normal_comments = driver.find_elements_by_css_selector('#new-posts div.tie-list div div.tie-bdy div p')
            top_comments = driver.find_elements_by_css_selector('#hot-posts div.tie-list div div.tie-bdy div p')
            top_comments_praise = driver.find_elements_by_css_selector('#hot-posts div.tie-list div div.tie-bdy'
                                                                       ' div div.tool-bar ul li a span')
            rec_di[str(now_page)]['normal_comments'] = [[''] for _ in range(len(normal_comments))]
            rec_di[str(now_page)]['top_comments'] = [[''] for _ in range(len(top_comments))]
            rec_di[str(now_page)]['top_comments_praises'] = [[''] for _ in range(len(top_comments))]

            for i, comment_selector in enumerate(normal_comments):
                rec_di[str(now_page)]['normal_comments'][i] = comment_selector.text
            for i, comment_selector in enumerate(top_comments):
                rec_di[str(now_page)]['top_comments'][i] = comment_selector.text
            for i, praise_selector in enumerate(top_comments_praise):
                if i % 2:
                    continue
                # tt = praise_selector.text
                # praise = int(praise_selector.text.replace('[', '').replace(']', ''))
                praise = praise_selector.get_attribute('data-post-up')
                rec_di[str(now_page)]['top_comments_praises'][int(i / 2)] = praise

            pic_id = denominator
            for _inner in range(denominator):
                # 上一张图片
                ele_to_hover = driver.find_element_by_css_selector('body div.gallery div.main div.photoarea')
                action = ActionChains(driver)
                action.move_to_element_with_offset(ele_to_hover, 10, 10)
                action.click()
                action.perform()
                pic_id -= 1
                sleep(0.2)

                b_img_desc_css_selector = 'div.picinfo-text-wrap div p span:nth-child(1)'
                a_img_desc_css_selector = 'div.picinfo-text-wrap div p span:nth-child(1)'
                b_img_css_selector = 'body div.gallery div.main div.photoarea div.photo-b img'
                a_img_css_selector = 'body div.gallery div.main div.photoarea div.photo-a img'
                if element_num(driver, b_img_css_selector):
                    img_desc = driver.find_element_by_css_selector(b_img_desc_css_selector).text
                    img_link = driver.find_element_by_css_selector(b_img_css_selector).get_attribute('src')
                else:
                    img_desc = driver.find_element_by_css_selector(a_img_desc_css_selector).text
                    img_link = driver.find_element_by_css_selector(a_img_css_selector).get_attribute('src')

                rec_di[str(now_page)]['img_desc'][pic_id] = img_desc
                rec_di[str(now_page)]['img_link'][pic_id] = img_link


            # 一个整页爬取完成 保存
            succ_cnt += 1
            succ = True
            if not succ_cnt % 25:
                print(succ_cnt)
                with open('save/{}_{}.json'.format(start_page, now_page), 'w') as f:
                    json.dump(rec_di, f)
            break
    except:
        print('exception at page {}'.format(now_page))
    if succ:
        print('*')
    else:
        print('.')
    now_page -= 1
