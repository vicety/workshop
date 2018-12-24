from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

def element_num(driver, css):
    return len(driver.find_elements_by_css_selector(css_selector=css))