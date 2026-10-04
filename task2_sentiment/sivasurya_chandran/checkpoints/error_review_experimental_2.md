# Error review — experimental_2

Worst slice: `contains_but_contrast` (error rate 0.047)

### confident_false_positive — test #29330
- true: **negative**, predicted: **positive** (P(pos)=0.9999)
- suggested error type: sarcasm / irony
- final error type: _
- proposed testable fix: _

> Wow love the place and everything is very clean and new!  Great place to come and relax worth a try!  Cheers,  Eric Van Nguyen Visited April 2012

### confident_false_positive — test #15988
- true: **negative**, predicted: **positive** (P(pos)=0.9997)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Local dining icons deserve their own criteria for recognition.  That said, adding sliced tomato, coleslaw and a pile of fries to a deli sandwich defines the Primanti as a Pittsburgh-area institution.  Their cheesesteaks will never threaten the best of Philly's, nor will their pastrami or corned beef be mistaken for New York deli; this only serves to make the Primanti style unique.  I find their meats to be generally good, and reasonably generous.  Tomato on a corned beef sandwich can only be regarded as an acquired taste.  Personally, I'm not a fan of loading a sandwich with fries, but again, 

### confident_false_positive — test #17407
- true: **negative**, predicted: **positive** (P(pos)=0.9995)
- suggested error type: truncation (key sentiment after max_seq_len cut-off)
- final error type: _
- proposed testable fix: _

> This was my second time dining at Company. And unfortunately, it will probably be the last. Maybe that's not fair. I'll explain.  I first dined at Company back in June. My party of six was seated by a waiter and looked after very attentively by three support servers clad all black pants, collared shirts and aprons. I felt a little under-dressed. The menu was upscale comfort food. My friend Katie gave me part of her delicious lobster BLT. My friend Colton gave me two spoonfuls of deliciously creamy mashed potatoes and the best creamed spinach I have ever had. I had the special that night, which

### confident_false_positive — test #12480
- true: **negative**, predicted: **positive** (P(pos)=0.9995)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> my husband had an omelette that was good. i had a blt, a little on the small side for $10, but bacon was great. Our server was awesome!

### confident_false_positive — test #163
- true: **negative**, predicted: **positive** (P(pos)=0.9992)
- suggested error type: implicit sentiment / domain-specific wording
- final error type: _
- proposed testable fix: _

> This place is defiantly a historic site! From the floor to the table tops to the bathrooms! I love learning about the history of how places came about. Mary the bartender gave me lots of info.......oldest bar, 1st one to let women in, same flooring, etc.   The food is frozen so if you're looking for fresh seafood this ain't it! I had one of their famous oysters.  Very good I highly recommend getting only 3 very filling. I had the special crab cakes & 2 sides. Imitation crab & lots of filling. So I don't need to tell you how nasty that was. My house made chips were good & cole slaw had very lit

### confident_false_negative — test #22807
- true: **positive**, predicted: **negative** (P(pos)=0.0001)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> EDIT: They really did change the service up since I last posted this.  Horrible service.  Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not give him attitude about it.'  Sorry Bella Notte, but we're not children. I don't care if you're working hard - it doesn't give you any excuse to disrespect your paying customers like that.

### confident_false_negative — test #20061
- true: **positive**, predicted: **negative** (P(pos)=0.0002)
- suggested error type: truncation (key sentiment after max_seq_len cut-off)
- final error type: _
- proposed testable fix: _

> Ever wonder what to do if you have lots of extra garbage or recyclables and either can't fit them all in your bins or missed bulk trash pickup day? Alternatively, are you looking for something just a little bit different to do on a lazy summer day?   Ok, ok, all kidding aside - you may find yourselves (as we did yesterday) with more broken-down boxes and odd-shaped trash (old floor lamps, etc.) than can fit in your bins. This waste facility allows Phoenix residents to dump bulk trash and recyclables for free once a month.   To get to the facility, drive down to 27th Avenue and Buckeye (aka \""

### confident_false_negative — test #26683
- true: **positive**, predicted: **negative** (P(pos)=0.0003)
- suggested error type: negation scope
- final error type: _
- proposed testable fix: _

> I've just been forced to concede that, despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review.

### confident_false_negative — test #11401
- true: **positive**, predicted: **negative** (P(pos)=0.0009)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Perhaps my expectations were too high because of all the hype I'd heard, but I have to say I was a little disappointed. We did a girls' night out and were so excited about this new great wine/chocolate combo thing, but after waiting forever for a table we waited forever for a waitress, who utimately messed up our order and brought us first the wrong food and then cold food. When they finally got the food right, it was actually pretty goos, but by that time were were kind of over it. As for dessert, we ended up going with these chocolate martinis that were pretty rockin. My suggestion is head t

### confident_false_negative — test #30793
- true: **positive**, predicted: **negative** (P(pos)=0.0009)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> This place is so much better since they changed owners.  My wife and I went when it was the old owners, it was terrible.  We waited forever and the food never came before we walked out.  People were served before us that walked in after and my wife actually got her soup before me and I sat and waited while they \""made more\"".  It was horrible.  Now its much better.  The staff are very friendly, they treat their customers very well and I have nothing but positive things to now say about this place.  Its much better with the new owners.

### near_threshold — test #37541
- true: **negative**, predicted: **positive** (P(pos)=0.5003)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Tried Rudy's for the first time tonight. Food was good, not great, and we weren't impressed with the \""share-a-table-with-your-neighor\"" atmosphere.  Eating with no plates was a bit odd as was the single slice of plain sandwich bread we received with all the meat we ordered. We aren't barbecue connesuers, though, so maybe this is normal? We'll definitely opt for other BBQ restaurants before returning here.

### near_threshold — test #26481
- true: **positive**, predicted: **negative** (P(pos)=0.4986)
- suggested error type: truncation (key sentiment after max_seq_len cut-off)
- final error type: _
- proposed testable fix: _

> My first experience with STK was at the NYC location, and I was hardly surprised when they decided to bring an STK to my current digs in Los Angeles -- a city that has its fair share of trendy restaurants that lack culinary inspiration. Suffices to say I was less than shocked when I heard the Cosmopolitan hotel in Las Vegas would also feature an STK restaurant.   My experiences at all three locations have been markedly different. The NYC location boasted good food and a classy atmosphere in the Meatpacking District (an area which wasn't hurting for culinary hot-spots at the time) and was able 

### near_threshold — test #4439
- true: **positive**, predicted: **negative** (P(pos)=0.4986)
- suggested error type: too little signal (very short review)
- final error type: _
- proposed testable fix: _

> mark & mercedes are a must listen to in the morning.

### near_threshold — test #18716
- true: **negative**, predicted: **positive** (P(pos)=0.5015)
- suggested error type: too little signal (very short review)
- final error type: _
- proposed testable fix: _

> where is this place?

### near_threshold — test #6825
- true: **negative**, predicted: **positive** (P(pos)=0.5019)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> I found water pooling around my fridge so went to Yelp for a recommendation. The fridge's 2-year warranty had just run out.   A very nice guy came on the early side of the 2-hour window. He was professional and respectful. However he did not find the true problem behind a leaky fridge. He said it was plumbing--luckily I have a home warranty and the cost for a plumber to come out was only $60. The leaking continued so I had a second plumber come out at no charge. Both plumbers demonstrated how this couldn't be a plumbing issue--the pipe was dry.   So I called E&J back and this time the guy did 

### slice:contains_but_contrast — test #9162
- true: **negative**, predicted: **positive** (P(pos)=0.5425)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Food is good but the portions are small for what you are paying. The only thing you are truly going to be fully satisfied with this place is the service and the ambience. If you are expecting to get hungry after you eat here, think again. You're probably gonna end up spending a fortune over here - just make sure you have at least $5 to head over to Planet Hollywood afterwards so you can get yourself something really filling - Earl of Sandwiches!

### slice:contains_but_contrast — test #25670
- true: **negative**, predicted: **positive** (P(pos)=0.8618)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Hate to be the bad review guy, but here goes. Dressed to the nines, the wife and I step out to dinner at this joint. We've been dining out in Vegas for over 20 years and this is a first. Upon being seated I am asked to remove my hat as they have a no-hat policy. I think to myself, must be fancy, as I remove my hat and begin to enjoy a great dinner. Food and service was very nice. As I am enjoying this succulent filet mignon, I notice that most other guests in this place are dressed for an outing to Wal-Mart. Shorts, dirty tee shirts from 80's rock concerts, etc.. I ask the waiter why they are 

### slice:contains_but_contrast — test #28735
- true: **positive**, predicted: **negative** (P(pos)=0.0719)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> Okay. Been there...done that. Quite possibly the greasiest pizza I've eaten since junior high cafeteria days. But it was good. If I was hankering for pizza, this would do!  Smiley face

### slice:contains_but_contrast — test #19109
- true: **negative**, predicted: **positive** (P(pos)=0.8475)
- suggested error type: mixed sentiment / contrast clause
- final error type: _
- proposed testable fix: _

> I have dined at this location a number of times, so I'm confident that this 2 star rating is as high as I can go. The lunch buffet is ample, with many vegetarian options, as well as a few meat choices. Almost without exception, however, every dish I've tried here is just too spicy. Now, I enjoy spicy food, and delight in the variety of chiles that are available, from the wonderful bite of a fresh green chile to the deeper and more complex flavors that dried chiles provide. What they use here at Tamarind is cayenne, which is a fairly one note chile that delivers little in the way of nuance and 

### slice:contains_but_contrast — test #30300
- true: **positive**, predicted: **negative** (P(pos)=0.1999)
- suggested error type: truncation (key sentiment after max_seq_len cut-off)
- final error type: _
- proposed testable fix: _

> Great Bao was closing their brick & motor location; which up til now I wasn't too keen on going due to its odd location inside a Salon...(I mean, how does that actually work? Food and Hair products just sound so wrong).  But seeing that they were closing (due to lease issues)  it was our last chance to support them. (they still have their food truck where you can have their most heavenly Baos).  We thought it would be cool as ice to take a stroll over to their soon-to-be defunct operation inside the 'house of hair'...  We were at a loss for words....wow.... It's really inside a Hair/Nails Salo

