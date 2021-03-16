---
title: "Growing teams - dealing with bottlenecks in test"
date: 2020-06-20T18:48:46-04:00
showDate: true
draft: false
tags: ["blog","story"]
mermaid: false
---

Testers can be seen as blocking the delivery of work -how many times have you heard 'its finished but still in test'?

Testers aren't blocking the work, work is still be treated as a waterfall process but with people/specialisms instead of scope:

![TeamDevFocus.png](/posts/TeamDevFocus.png)

In the above work is stacking up waiting to be tested. Developers are continuing to churn out work that needs to be tested but there's not much to show for it that has been deployed.

On the face of it an obvious solution may be that you need more testers, however what this may result in testers not having anything to do until developers have completed their part.

In my opinion, a better solution is to change the focus from developers developing and testers testing to: the team delivering value. The simplest way to do this is enforce a maxiumum number of work items in progress (WIP) in a given phase. If we say that there can be no more than 3 WIP in test/QA (Ready for test, in test) then once that limit is exceeded resource is shifted  within the team  to increase the throughput:

![TeamTestFocus.png](/posts/TeamTestFocus.png)

'Move' the developers to the test phase. I say move, they're still in the same seats doing similar work but now their focus is making sure things work rather than changing/adding things. Thereby enabling more value to be delivered.

![TeamTestFocusResultl.png](/posts/TeamTestFocusResult.png)

The advantages don't stop there:

-   Developers become better engineers, testing their work more thoroughly as they will be picking it up if a backlog is created in test (a lot of backlogs in test occur where something doesn't work as per the story resulting in a lot of costly back and forth in the QA phase)
-   Changes are more likely to work first time (due to better testing by developers) which increases the velocity of test
-   Testers are freed to do what the excel at - fringe scenarios and edge cases, thinking like a user
-   Developers are more likely to have the skills to and create automation tests - further increases the ability to test quickly, even better these can be added to build pipelines to super increase your test ability
-   And so on...

'_But now I'm not doing as much work in a team_' -if your message of the success is the delivering value (it should be) then you were'nt going to deliver those items anyway as they couldn't all be tested in time. They will have rolled over to the next sprint/period