# -*- coding: utf-8 -*-
from translate import Translator

translator= Translator(to_lang="en", from_lang="zh")

ch_content = "Tim Hortons从未间断过对于各项体育事业的支持，渐渐形成了“顾客买高热量快餐长肉，店家拿钱创造运动机会给大家减肥”的良好循环。不过，只是作为赞助商的话，大家还不至于对它投入那么大的感情。Tim Hortons相当高明与暖心的一手，就是对孩子们追求体育梦想的帮助，无论孩子出身贫穷，还是家境富有。它建立起的Timbits系统，几乎占据了所有冰球明星的童年。冰球之于加拿大，正如棒球之于美国，乒乓之于中国。受人追捧的国球明星，与自己天天享用的咖啡的连结，毫无疑问促成了感情上螺旋般的盘绕上升。冰球与Tim Hortons，也就更深地烙进了加拿大人心里。Tim Hortons会抓人心这一点，还体现在了它与加拿大军队的合作上。将分店开到境外军事基地，为背井离乡的官兵送温暖献爱心，Tim Hortons自然而然地成了军人眼里家的象征，也是亲人心中思念的寄托。在加拿大生活过的小伙伴们知道，每一年的11月11日，中国人普遍关心着光棍节这个问题，而加拿大人却是戴起罂粟花纪念国殇日。（这倒不是说中国人不关心国家军事，只是大陆并没有国殇日这个概念。毕竟一战我们也不怎么核心。）Tim Hortons是2004年加拿大造币总公司Canadian Mint发售的国殇日纪念币官方发售商，这心系祖国的商标形象，就在加拿大人买咖啡的分分钟之间塑造起来了。"
# ch_content = "从未间断过对于各项体育事业的支持渐渐形成了"

content = translator.translate(ch_content)

print content