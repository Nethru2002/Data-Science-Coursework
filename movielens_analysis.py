from pathlib import Path
import json, zipfile, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "ml-25m.zip"
DATA = ROOT / "ml-25m"
OUT = ROOT / "movielens_outputs"
OUT.mkdir(exist_ok=True)
np.random.seed(42)

if not DATA.exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(ROOT)

ratings = pd.read_csv(DATA / "ratings.csv", dtype={"userId":"int32","movieId":"int32","rating":"float32","timestamp":"int64"})
movies = pd.read_csv(DATA / "movies.csv", dtype={"movieId":"int32","title":"string","genres":"string"})
tags = pd.read_csv(DATA / "tags.csv", dtype={"userId":"int32","movieId":"int32","tag":"string","timestamp":"int64"})

stats = {}
stats["ratings_rows"] = int(len(ratings)); stats["movies_rows"] = int(len(movies)); stats["tags_rows"] = int(len(tags))
stats["users"] = int(ratings.userId.nunique()); stats["rated_movies"] = int(ratings.movieId.nunique())
stats["rating_duplicates_exact"] = int(ratings.duplicated().sum())
stats["rating_duplicate_user_movie"] = int(ratings.duplicated(["userId","movieId"]).sum())
stats["ratings_missing"] = {c:int(v) for c,v in ratings.isna().sum().items()}
stats["movies_missing"] = {c:int(v) for c,v in movies.isna().sum().items()}
stats["tags_missing"] = {c:int(v) for c,v in tags.isna().sum().items()}
stats["rating_mean"] = float(ratings.rating.mean()); stats["rating_median"] = float(ratings.rating.median()); stats["rating_std"] = float(ratings.rating.std())
stats["rating_min"] = float(ratings.rating.min()); stats["rating_max"] = float(ratings.rating.max())
stats["rating_counts"] = {str(k):int(v) for k,v in ratings.rating.value_counts().sort_index().items()}
stats["high_rating_pct"] = float((ratings.rating >= 4).mean()*100)
stats["timestamp_min"] = pd.to_datetime(ratings.timestamp.min(), unit="s", utc=True).isoformat()
stats["timestamp_max"] = pd.to_datetime(ratings.timestamp.max(), unit="s", utc=True).isoformat()

movies["releaseYear"] = pd.to_numeric(movies.title.str.extract(r"\((\d{4})\)\s*$", expand=False), errors="coerce")
stats["missing_release_year"] = int(movies.releaseYear.isna().sum())
stats["no_genres_listed"] = int((movies.genres == "(no genres listed)").sum())

movie_counts = ratings.groupby("movieId", sort=False).size().sort_values(ascending=False)
user_counts = ratings.groupby("userId", sort=False).size().sort_values(ascending=False)
stats["movie_count_median"] = float(movie_counts.median()); stats["movie_count_p99"] = float(movie_counts.quantile(.99)); stats["movie_count_max"] = int(movie_counts.iloc[0])
stats["user_count_median"] = float(user_counts.median()); stats["user_count_p99"] = float(user_counts.quantile(.99)); stats["user_count_max"] = int(user_counts.iloc[0])
for frac in [.01,.05,.10,.20]:
    n=max(1,int(np.ceil(len(movie_counts)*frac)))
    stats[f"top_{int(frac*100)}pct_movie_share"] = float(movie_counts.iloc[:n].sum()/len(ratings)*100)

genre = movies[["movieId","genres"]].copy()
genre["genre"] = genre.genres.str.split("|")
genre = genre.explode("genre")
gr = ratings[["movieId","rating"]].merge(genre[["movieId","genre"]], on="movieId", how="left")
genre_stats = gr.groupby("genre").rating.agg(["count","mean"]).sort_values("count",ascending=False)
stats["genre_stats"] = {str(i):{"count":int(r["count"]),"mean":float(r["mean"])} for i,r in genre_stats.iterrows()}
del gr

sns.set_theme(style="whitegrid")
fig,ax=plt.subplots(figsize=(8,5)); rc=ratings.rating.value_counts().sort_index(); ax.bar(rc.index.astype(str),rc.values,color="#4C78A8")
ax.set(title="Distribution of 25,000,095 MovieLens ratings",xlabel="Rating",ylabel="Number of ratings"); fig.tight_layout(); fig.savefig(OUT/"fig1_rating_distribution.png",dpi=180); plt.close(fig)

years=pd.to_datetime(ratings.timestamp,unit="s",utc=True).dt.year.value_counts().sort_index()
stats["annual_counts"]={str(k):int(v) for k,v in years.items()}
fig,ax=plt.subplots(figsize=(10,5)); ax.plot(years.index,years.values,marker="o",lw=2,color="#F58518"); ax.set(title="Ratings recorded by calendar year",xlabel="Year",ylabel="Ratings"); fig.tight_layout(); fig.savefig(OUT/"fig2_ratings_by_year.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5)); ax.loglog(np.arange(1,len(movie_counts)+1),movie_counts.values,color="#54A24B"); ax.set(title="Movie popularity is strongly long-tailed",xlabel="Movie popularity rank (log scale)",ylabel="Number of ratings (log scale)"); fig.tight_layout(); fig.savefig(OUT/"fig3_movie_long_tail.png",dpi=180); plt.close(fig)

gs=genre_stats[genre_stats["count"]>=100000].sort_values("mean")
fig,ax=plt.subplots(figsize=(9,6)); sc=ax.scatter(gs["mean"],gs["count"],s=np.sqrt(gs["count"])*.25,c=gs["mean"],cmap="viridis")
for name,row in gs.iterrows(): ax.annotate(name,(row["mean"],row["count"]),fontsize=8,xytext=(3,2),textcoords="offset points")
ax.set_yscale("log"); ax.set(title="Genre rating volume versus average rating",xlabel="Mean rating",ylabel="Rating count (log scale)"); fig.colorbar(sc,ax=ax,label="Mean rating"); fig.tight_layout(); fig.savefig(OUT/"fig4_genre_volume_mean.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5)); ax.hist(user_counts.values,bins=np.logspace(np.log10(user_counts.min()),np.log10(user_counts.max()),50),color="#E45756"); ax.set_xscale("log"); ax.set(title="User activity distribution",xlabel="Ratings per user (log scale)",ylabel="Users"); fig.tight_layout(); fig.savefig(OUT/"fig5_user_activity.png",dpi=180); plt.close(fig)

cutoff=int(ratings.timestamp.quantile(.80)); stats["temporal_cutoff"] = pd.to_datetime(cutoff,unit="s",utc=True).isoformat()
train=ratings[ratings.timestamp<=cutoff].copy(); test=ratings[ratings.timestamp>cutoff].copy()
stats["train_rows_full"]=int(len(train)); stats["test_rows_full"]=int(len(test))
global_mean=float(train.rating.mean())
uagg=train.groupby("userId").rating.agg(userSum="sum",userCount="size")
magg=train.groupby("movieId").rating.agg(movieSum="sum",movieCount="size")

def features(df, training_rows=False):
    x=df[["userId","movieId","timestamp","rating"]].join(uagg,on="userId").join(magg,on="movieId").merge(movies[["movieId","releaseYear"]],on="movieId",how="left")
    dt=pd.to_datetime(x.timestamp,unit="s",utc=True)
    x["ratingYear"]=dt.dt.year.astype("int16"); x["ratingMonth"]=dt.dt.month.astype("int8")
    x["userCount"]=x.userCount.fillna(0); x["movieCount"]=x.movieCount.fillna(0)
    lam=10.0
    if training_rows:
        x["userMean"]=(x.userSum-x.rating+lam*global_mean)/(x.userCount-1+lam)
        x["movieMean"]=(x.movieSum-x.rating+lam*global_mean)/(x.movieCount-1+lam)
    else:
        x["userMean"]=(x.userSum+lam*global_mean)/(x.userCount+lam)
        x["movieMean"]=(x.movieSum+lam*global_mean)/(x.movieCount+lam)
    x["userMean"]=x.userMean.fillna(global_mean); x["movieMean"]=x.movieMean.fillna(global_mean)
    median_year=float(movies.releaseYear.median()); x["releaseYear"]=x.releaseYear.fillna(median_year)
    x["movieAge"]=np.maximum(0,x.ratingYear-x.releaseYear)
    x["logUserCount"]=np.log1p(x.userCount); x["logMovieCount"]=np.log1p(x.movieCount)
    return x

train_s=train.sample(n=min(2_000_000,len(train)),random_state=42)
test_s=test.sample(n=min(1_000_000,len(test)),random_state=42)
tr=features(train_s,training_rows=True); te=features(test_s,training_rows=False)
cols=["userMean","movieMean","logUserCount","logMovieCount","releaseYear","movieAge","ratingMonth"]
Xtr=tr[cols].astype("float32"); ytr=tr.rating.astype("float32"); Xte=te[cols].astype("float32"); yte=te.rating.astype("float32")

def evaluate(name,pred,seconds):
    pred=np.clip(pred,.5,5.0)
    return {"model":name,"RMSE":float(mean_squared_error(yte,pred)**.5),"MAE":float(mean_absolute_error(yte,pred)),"R2":float(r2_score(yte,pred)),"train_seconds":float(seconds)}

order=np.argsort(tr.timestamp.to_numpy()); split=int(len(order)*.8); fit_idx=order[:split]; val_idx=order[split:]
tuning=[]
best_ridge=(None,1e9)
for alpha in [1.0,10.0,100.0]:
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)); m.fit(Xtr.iloc[fit_idx],ytr.iloc[fit_idx]); p=np.clip(m.predict(Xtr.iloc[val_idx]),.5,5)
    score=float(mean_squared_error(ytr.iloc[val_idx],p)**.5); tuning.append({"model":"Ridge","parameters":f"alpha={alpha}","validation_RMSE":score})
    if score<best_ridge[1]: best_ridge=(alpha,score)
best_hgb=(None,1e9)
for leaves,lr in [(15,.05),(31,.05),(31,.08),(63,.05)]:
    m=HistGradientBoostingRegressor(max_iter=200,learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=1.0,random_state=42,early_stopping=True,validation_fraction=.1)
    m.fit(Xtr.iloc[fit_idx],ytr.iloc[fit_idx]); p=np.clip(m.predict(Xtr.iloc[val_idx]),.5,5)
    score=float(mean_squared_error(ytr.iloc[val_idx],p)**.5); tuning.append({"model":"HGB","parameters":f"leaves={leaves}, lr={lr}","validation_RMSE":score})
    if score<best_hgb[1]: best_hgb=((leaves,lr),score)
models=[]
t=time.time(); ridge=make_pipeline(StandardScaler(),Ridge(alpha=best_ridge[0])); ridge.fit(Xtr,ytr); models.append(evaluate("Ridge regression",ridge.predict(Xte),time.time()-t))
leaves,lr=best_hgb[0]
t=time.time(); hgb=HistGradientBoostingRegressor(max_iter=200,learning_rate=lr,max_leaf_nodes=leaves,l2_regularization=1.0,random_state=42,early_stopping=True,validation_fraction=.1); hgb.fit(Xtr,ytr); models.append(evaluate("Histogram gradient boosting",hgb.predict(Xte),time.time()-t))
models.insert(0,evaluate("Global-mean baseline",np.full(len(yte),global_mean),0))
models.insert(1,evaluate("Movie-mean baseline",te.movieMean.to_numpy(),0))
stats["model_results"]=models
stats["tuning_results"]=tuning
stats["selected_ridge_alpha"]=best_ridge[0]; stats["selected_hgb"]={"max_leaf_nodes":leaves,"learning_rate":lr}
stats["model_train_sample"]=int(len(tr)); stats["model_test_sample"]=int(len(te)); stats["global_train_mean"]=global_mean
stats["unseen_test_users_pct"]=float((te.userCount==0).mean()*100); stats["unseen_test_movies_pct"]=float((te.movieCount==0).mean()*100)

md=pd.DataFrame(models)
fig,ax=plt.subplots(figsize=(9,5)); bars=ax.bar(md.model,md.RMSE,color=["#BAB0AC","#9D755D","#4C78A8","#59A14F"]); ax.bar_label(bars,fmt="%.4f"); ax.set(title="Temporal hold-out model comparison",ylabel="RMSE (lower is better)"); ax.tick_params(axis="x",rotation=15); fig.tight_layout(); fig.savefig(OUT/"fig6_model_rmse.png",dpi=180); plt.close(fig)

corr=tr[["rating"]+cols].corr(method="spearman")["rating"].drop("rating").sort_values(key=abs,ascending=False)
stats["spearman_correlations"]={k:float(v) for k,v in corr.items()}

with open(OUT/"results.json","w",encoding="utf-8") as f: json.dump(stats,f,indent=2)
pd.DataFrame(models).to_csv(OUT/"model_results.csv",index=False)
print(json.dumps({"stats_file":str(OUT/"results.json"),"models":models},indent=2))
