(function waitAndPad(){
  const tablist = document.querySelector('div[data-testid="stTabs"] div[role="tablist"]');
  if(tablist){
    if(!tablist.querySelector('.__tablist-spacer')){
      const s = document.createElement('div');
      s.className = '__tablist-spacer';
      s.style.width = '24px';
      s.style.flex = '0 0 24px';
      s.style.height = '1px';
      s.style.display = 'block';
      tablist.appendChild(s);
      tablist.style.paddingRight = '24px';
    }
    return;
  }
  setTimeout(waitAndPad, 150);
})();
