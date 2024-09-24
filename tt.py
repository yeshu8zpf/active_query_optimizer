import torch

a = torch.randint(0, 10, (5,))
b = torch.randint(0, 10, (5,))
c = torch.max(a, b)
pass

"SELECT COUNT(*) FROM posthistory AS ph, postlinks AS pl, comments AS c, votes AS v, posts AS p WHERE\
c.postid = pl.postid AND ph.postid = pl.postid AND pl.postid = v.postid AND p.id = pl.postid\
AND ph.PostId > 39972 AND p.OwnerUserId < 4719 AND v.UserId >= 30865 AND p.PostTypeId != 6;"
