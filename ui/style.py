"""Visual styling shared by all pages (static CSS, no user content)."""

import streamlit as st

GREEN = "#2E7D32"

CSS = """
<style>
/* tighter page top */
.block-container {padding-top: 3.6rem; max-width: 1200px;}

/* hero banner */
.tmp-hero {
  background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 45%, #43A047 100%);
  border-radius: 18px; padding: 2.4rem 2.4rem 2.1rem; color: #fff;
  box-shadow: 0 10px 30px rgba(27,94,32,.25); margin-bottom: 1.4rem;
  position: relative; overflow: hidden;
}
.tmp-hero:after {               /* soft decorative ring */
  content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
  border-radius: 50%; border: 36px solid rgba(255,255,255,.08);
}
.tmp-hero h1 {color: #fff; font-size: 2.6rem; font-weight: 800; margin: 0; padding: 0;
              letter-spacing: -.02em;}
.tmp-hero p {color: rgba(255,255,255,.92); font-size: 1.15rem; margin: .4rem 0 1.1rem;}
.tmp-chip {display: inline-block; background: rgba(255,255,255,.16); color: #fff;
           border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
           padding: .22rem .75rem; margin: 0 .35rem .35rem 0; font-size: .82rem;}

/* stats strip */
.tmp-stats {display: flex; gap: .8rem; flex-wrap: wrap; margin: .2rem 0 1.2rem;}
.tmp-stat {flex: 1 1 140px; background: #F1F8F2; border-radius: 14px; padding: .9rem 1rem;}
.tmp-stat b {display: block; font-size: 1.5rem; color: #1B5E20; line-height: 1.2;}
.tmp-stat span {font-size: .85rem; color: #4A4A4A;}

/* workflow chips */
.tmp-flow {display: flex; align-items: center; flex-wrap: wrap; gap: .4rem; margin: .3rem 0 .6rem;}
.tmp-step {background: #fff; border: 1px solid #C8E6C9; color: #1B5E20; border-radius: 10px;
           padding: .35rem .8rem; font-size: .9rem; font-weight: 600;}
.tmp-arrow {color: #81C784; font-weight: 700;}

/* page headers */
.tmp-title {font-size: 1.9rem; font-weight: 800; color: #1B5E20; margin: 0;
            letter-spacing: -.01em;}
.tmp-sub {color: #5F6368; margin: .15rem 0 1rem; font-size: 1rem;}

/* bordered cards: gentle hover lift */
div[data-testid="stVerticalBlockBorderWrapper"] {transition: box-shadow .2s, transform .2s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  box-shadow: 0 8px 22px rgba(46,125,50,.14); transform: translateY(-2px);
}
/* hero layout with molecule art + entrance animation */
@keyframes tmpFade {from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;}}
.tmp-hero {display: flex; align-items: center; gap: 1.5rem; animation: tmpFade .6s ease-out;}
.tmp-hero-text {flex: 1 1 60%; z-index: 1;}
.tmp-hero-art {flex: 0 0 34%; max-width: 360px; opacity: .9; z-index: 1;
               filter: drop-shadow(0 4px 12px rgba(0,0,0,.15));}
@media (max-width: 800px) {.tmp-hero-art {display: none;}}
.tmp-stat {animation: tmpFade .6s ease-out both;}
.tmp-stat i {font-style: normal; font-size: 1.3rem; float: right; opacity: .8;}

/* molecule cards */
.tmp-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
           gap: .9rem; margin: .6rem 0 1rem;}
.tmp-card {position: relative; background: #fff; border: 1px solid #E3EDE4; border-radius: 16px;
           padding: .7rem .8rem .8rem; transition: box-shadow .2s, transform .2s;
           animation: tmpFade .5s ease-out both;}
.tmp-card:hover {box-shadow: 0 10px 24px rgba(46,125,50,.16); transform: translateY(-3px);}
.tmp-query {background: linear-gradient(160deg, #F1F8F2, #E3F2E5); border: 2px dashed #81C784;}
.tmp-mol {height: 150px; display: flex; align-items: center; justify-content: center;}
.tmp-mol svg {max-height: 100%;}
.tmp-card-title {font-weight: 700; font-size: .92rem; margin-top: .3rem;}
.tmp-card-title a {color: #1B5E20; text-decoration: none;}
.tmp-card-title a:hover {text-decoration: underline;}
.tmp-card-name {color: #5F6368; font-size: .8rem; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis;}
.tmp-score {position: absolute; top: .6rem; right: .6rem; border-radius: 999px;
            padding: .1rem .55rem; font-size: .8rem; font-weight: 700; color: #fff; z-index: 1;}
.tmp-high {background: #2E7D32;} .tmp-mid {background: #F9A825;} .tmp-low {background: #9E9E9E;}

/* page header banner */
.tmp-pagehead {display: flex; align-items: center; gap: 1rem; padding: 1.1rem 1.4rem;
               margin-bottom: 1.2rem; border-radius: 16px;
               background: linear-gradient(120deg, #E8F5E9 0%, #F1F8F2 60%, #FFFFFF 100%);
               border: 1px solid #DCEDDF; animation: tmpFade .5s ease-out;}
.tmp-ph-icon {font-size: 2rem; width: 3.2rem; height: 3.2rem; display: flex;
              align-items: center; justify-content: center; border-radius: 14px;
              background: linear-gradient(135deg, #2E7D32, #66BB6A);
              box-shadow: 0 6px 16px rgba(46,125,50,.25);}
.tmp-pagehead h2 {margin: 0; padding: 0; font-size: 1.7rem; font-weight: 800; color: #1B5E20;}
.tmp-pagehead p {margin: .1rem 0 0; color: #5F6368;}

/* query preview card */
.tmp-preview {background: #fff; border: 1px solid #E3EDE4; border-radius: 16px;
              padding: .6rem .8rem; animation: tmpFade .4s ease-out;}
.tmp-preview .tmp-mol {height: 170px;}
.tmp-props {display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .3rem;}
.tmp-prop {background: #F1F8F2; color: #1B5E20; border-radius: 8px; padding: .12rem .5rem;
           font-size: .78rem; font-weight: 600;}
.tmp-prop.warn {background: #FFF8E1; color: #8D6E00;}

/* primary buttons */
button[data-testid="stBaseButton-primary"], button[kind="primary"] {
  background: linear-gradient(120deg, #1B5E20, #43A047) !important; border: none !important;
  box-shadow: 0 6px 16px rgba(46,125,50,.25); transition: transform .15s, box-shadow .15s;}
button[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 8px 20px rgba(46,125,50,.35);}

/* workflow inside the hero */
.tmp-hero-flow {display: flex; flex-wrap: wrap; align-items: center; gap: .15rem;}
.tmp-hero-flow .tmp-chip {margin: 0;}
.tmp-hero-arrow {color: rgba(255,255,255,.75); margin: 0 .3rem; font-weight: 700;}
.tmp-powered {text-align: center; color: #9E9E9E; font-size: .82rem; margin-top: 1.6rem;}

/* metric numbers in brand colour */
div[data-testid="stMetricValue"] {color: #1B5E20;}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "🧬") -> None:
    """Branded page banner: icon tile, title and a one-line subtitle."""
    st.markdown(f'<div class="tmp-pagehead"><div class="tmp-ph-icon">{icon}</div>'
                f'<div><h2>{title}</h2><p>{subtitle}</p></div></div>',
                unsafe_allow_html=True)
