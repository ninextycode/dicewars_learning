var canvas, stage;
var builder;
var touchdev = false;

// イベント関数
var timer_func = new Function();	timer_func = null;
var click_func = new Function();	click_func = null;
var move_func = new Function();		move_func = null;
var release_func = new Function();	release_func = null;
var waitcount=0;
var stat=0;

// ゲームオブジェクト
var game = new Game();

// 表示位置
var org = {view_w:840,view_h:840,cel_w:27,cel_h:18,ypos_mes:688,ypos_arm:770};	// オリジナルサイズ
var nume = 1;
var deno = 1;
var view_w,view_h;
var cel_w,cel_h;	// カードの大きさと
var ypos_mes;		// メッセージ、戦闘ダイスの位置
var ypos_arm;		// 各軍のステータス表示位置
var dot;			// 1ドットの大きさ

// セル描画位置
var cpos_x = new Array();
var cpos_y = new Array();

// スプライト
var spr = new Array();

// スプライト番号
var sn_area = 0;
var sn_from = 0;	// 攻撃元エリアのスプライト番号
var sn_to = 0;	// 攻撃先エリアのスプライト番号
var sn_dice = 0;
var sn_info = 0;
var sn_ban = 0;
var sn_player = 0;
var sn_battle = 0;
var sn_supply = 0;
var sn_gameover = 0;
var sn_win = 0;
var sn_title = 0;
var sn_pmax = 0;
var sn_load = 0;
var sn_mes = 0;
var sn_btn = 0;
var sn_max = 0;	// 最大数

var prio = new Array();		// エリアダイスの表示順
var an2sn = new Array();	// エリア番号からダイススプライト番号を返す

// ボタン
var bmax = 0;
var activebutton = -1;
var btn_func = new Array();

// バトルクラス
var Battle = function(){
	this.dn = 0;	// ダイス番号(止めるべき位置)
	this.arm = 0;	// ダイス色
	this.dmax = 0;	// ダイス数
	this.deme = [0,0,0,0,0,0,0,0];
	this.sum = 0;
	this.fin = [0,0,0,0,0,0,0,0];	// 終了フラグ
	this.usedice = [0,1,2,3,4,5,6,7];	// 使うダイス
}
var battle = new Array();
var bturn = 0;	// バトル用ターン

// 履歴の再生用
var replay_c=0;

// サウンド関係
var soundon = true;
var manifest = [
	{"src":"./DICEWARS_files/sound/button.wav",	"id":"snd_button"},
	{"src":"./DICEWARS_files/sound/clear.wav",		"id":"snd_clear"},
	{"src":"./DICEWARS_files/sound/click.wav",		"id":"snd_click"},
	{"src":"./DICEWARS_files/sound/dice.wav",		"id":"snd_dice"},
	{"src":"./DICEWARS_files/sound/fail.wav",		"id":"snd_fail"},
	{"src":"./DICEWARS_files/sound/myturn.wav",	"id":"snd_myturn"},
	{"src":"./DICEWARS_files/sound/over.wav",		"id":"snd_over"},
	{"src":"./DICEWARS_files/sound/success.wav",	"id":"snd_success"}
];

// 縮尺に合わせてリサイズ
function resize(n){
	return n*nume/deno;
}

// 起動
window.addEventListener("load", init);
function init() {
	var i,j,c,n;

	canvas = document.getElementById("myCanvas");
	stage = new createjs.Stage(canvas);
	
	if(createjs.Touch.isSupported() == true) {
	   createjs.Touch.enable(stage);
	   touchdev = true;
	}
//	   touchdev = true;
	if( touchdev ){
		soundon = false;
	}
	// 表示位置
	var iw = window.innerWidth;
	var ih = window.innerHeight;
	if( iw/org.view_w<ih/org.view_h ){
		nume = iw;
		deno = org.view_w;
	}else{
		nume = ih;
		deno = org.view_h;
	}
	view_w = Math.floor(org.view_w*nume/deno);
	view_h = Math.floor(org.view_h*nume/deno);
	stage.canvas.width = view_w;
	stage.canvas.height = view_h;
	cel_w = org.cel_w*nume/deno;
	cel_h = org.cel_h*nume/deno;
	ypos_mes = org.ypos_mes*nume/deno;
	ypos_arm = org.ypos_arm*nume/deno;
	dot = 1*nume/deno;
	for( i=0; i<2; i++ ) battle[i] = new Battle();

	// スプライト番号
	var sn = 0;

	// セルの位置
	c=0;
	for( i=0; i<game.YMAX; i++ ){
		for( j=0; j<game.XMAX; j++ ){
			cpos_x[c] = j*cel_w;
			if( i%2 ) cpos_x[c] += cel_w/2;
			cpos_y[c] = i*cel_h;
		c++;
		}
	}
	
	// エリア描画 +2 (攻撃元と攻撃先)
	sn_area = sn;
	for( i=0; i<game.AREA_MAX+2; i++ ){
		spr[sn] = new createjs.Shape();
		spr[sn].x = view_w/2-game.XMAX*cel_w/2-cel_w/4;
		spr[sn].y = 50*nume/deno;
		stage.addChild(spr[sn]);
		sn++;
	}
	sn_from = sn_area + game.AREA_MAX;	// 攻撃元エリアのスプライト番号
	sn_to = sn_area + game.AREA_MAX+1;	// 攻撃先エリアのスプライト番号
	
	// エリアダイス
	sn_dice = sn;
	builder = new createjs.SpriteSheetBuilder();
	var mc = new lib.areadice();
	var rect = new createjs.Rectangle(0,0,80,100);
	builder.addMovieClip(mc, rect, nume/deno);
	var spritesheet = builder.build();
	for( i=0; i<game.AREA_MAX; i++ ){
		spr[sn] = new createjs.Sprite(spritesheet);
		stage.addChild(spr[sn]);
		sn++;
	}
	// エリアダイス表示順
	for( i=0; i<game.AREA_MAX; i++ ){
		prio[i] = new Object();
		prio[i].an = i;
		prio[i].cpos = 0;	// 後で使う
	}
	
	// マップ以外のスプライト番号(一括で消すため)
	sn_info = sn;
	
	// プレイヤー状態
	sn_ban = sn;
	spr[sn] = new lib.mc();
	stage.addChild(spr[sn]);
	spr[sn].scaleX = nume/deno;
	spr[sn].scaleY = nume/deno;
	sn++;
	sn_player = sn;
	for( i=0; i<8; i++ ){
		var pd = new lib.mc();
		pd.scaleX = pd.scaleY = 0.12;
		pd.x = -22;
		pd.y = 0;
		spr[sn] = new createjs.Container();
		spr[sn].addChildAt(pd,0);
		var txt = new createjs.Text("", "32px Anton", "Black")
		txt.textBaseline = "middle";
		txt.x = 5;
		spr[sn].addChildAt(txt,1);
		var txt2 = new createjs.Text("", "16px Anton", "Black")
		txt2.textBaseline = "middle";
		txt2.x = 5;
		txt2.y = 28;
		spr[sn].addChildAt(txt2,2);
		stage.addChild(spr[sn]);
		spr[sn].scaleX = nume/deno;
		spr[sn].scaleY = nume/deno;
		sn++;
	}
	
	// バトルダイス
	sn_battle = sn;
	spr[sn] = new createjs.Container();
	spr[sn].y = ypos_mes;
	spr[sn].x = view_w/2;
	spr[sn].scaleX = spr[sn].scaleY = nume/deno;
	var bgshape = new createjs.Shape();
	bgshape.graphics.beginFill("rgba(255,255,255,0.8)").drawRect(-org.view_w/2,-50,org.view_w,360);
	spr[sn].addChild(bgshape);
	for( i=0; i<2; i++ ){
		for( j=0; j<8; j++ ){
			var bs = new lib.mc();
			bs.scaleX = bs.scaleY = 0.15;
			bs.name = "s"+i+j;
			spr[sn].addChild(bs);
		}
		for( j=0; j<8; j++ ){
			var bd = new lib.mc();
			bd.scaleX = bd.scaleY = 0.15;
			bd.name = "d"+i+j;
			spr[sn].addChild(bd);
		}
		var txt = new createjs.Text("37", "80px Anton", "Black")
		txt.textBaseline = "middle";
		txt.textAlign = "center";
		txt.name = "n"+i;
		spr[sn].addChild(txt);
	}
	stage.addChild(spr[sn]);
	sn++;
	
	// 供給ダイス
	sn_supply = sn;
	spr[sn] = new createjs.Container();
	spr[sn].y = ypos_mes;
	spr[sn].x = view_w/2;
	spr[sn].scaleX = spr[sn].scaleY = nume/deno;
	for( i=0; i<game.STOCK_MAX; i++ ){
		var sd = new lib.mc();
		var w = 40;
		sd.x = -(6.5*w)+Math.floor(i/4)*w -(i%4)*w*0.5;
		sd.y = -w*0.7+Math.floor(i%4)*w/2;
		sd.gotoAndStop("d00");
		sd.scaleX = sd.scaleY = 0.1;
		spr[sn].addChildAt(sd,i);
	}
	stage.addChild(spr[sn]);
	sn++;

	// GAMEOVER
	sn_gameover = sn;
	spr[sn] = new createjs.Container();
	spr[sn].x = view_w/2;
	spr[sn].y = view_h/2;
	spr[sn].scaleX = spr[sn].scaleY = nume/deno;
	var goshape = new createjs.Shape();
	goshape.graphics.beginFill("#000000").drawRect(-org.view_w/2+10,-180,org.view_w-20,360);
	goshape.name = "bg";
	spr[sn].addChild(goshape);
	var gotext = new createjs.Text("G A M E O V E R", "80px Anton", "White")
	gotext.textBaseline = "middle";
	gotext.textAlign = "center";
	gotext.name = "mes";
	spr[sn].addChild(gotext);
	stage.addChild(spr[sn]);
	sn++;
	
	// YOU WIN
	sn_win = sn;
	spr[sn] = new lib.mc();
	spr[sn].scaleX = spr[sn].scaleY = nume/deno;
	stage.addChild(spr[sn]);
	sn++;
	
	// タイトル画面
	sn_title = sn;
	spr[sn] = new lib.mc();
	spr[sn].scaleX = spr[sn].scaleY = nume/deno;
	stage.addChild(spr[sn]);
	sn++;
	
	// プレイヤー数設定
	sn_pmax = sn;
	spr[sn] = new createjs.Container();
	for( i=0; i<7; i++ ){
		var ptxt = new createjs.Text((i+2)+" players",Math.floor(32*nume/deno)+"px Anton", "#aaaaaa");
		ptxt.name = "p"+i;
		ptxt.x = view_w/2 -280*nume/deno + Math.floor(i%4)*(180*nume/deno);
		ptxt.y = view_h*0.8 + Math.floor(i/4)*(60*nume/deno);
		ptxt.textAlign = "center";
		ptxt.textBaseline = "middle";
		spr[sn].addChild(ptxt);
	}
	stage.addChild(spr[sn]);
	sn++;
	
	// Loading用(web フォントを読んでおくため)
	sn_load = sn;
	spr[sn] = new createjs.Text("Now loading...", Math.floor(24*nume/deno)+"px Anton", "#000000");
	stage.addChild(spr[sn]);
	sn++;

	// 汎用メッセージ
	sn_mes = sn;
	spr[sn] = new createjs.Text("Now loading...", Math.floor(30*nume/deno)+"px Roboto", "#000000");
	spr[sn].textAlign = "center";
	spr[sn].textBaseline = "middle";
	stage.addChild(spr[sn]);
	sn++;
	
	// ボタン
	var btxt = ["START","TOP PAGE","YES","NO","END TURN","TITLE","HISTORY"];
	bmax = btxt.length;
	sn_btn = sn;
	for( i=0; i<bmax; i++ ){
		var bt = new lib.mc();
		spr[sn] = new createjs.Container();
		bt.gotoAndStop("btn");
		spr[sn].addChildAt(bt,0);
		var txt = new createjs.Text(btxt[i], "32px Anton", "Black")
		txt.textAlign = "center";
		txt.textBaseline = "middle";
		spr[sn].addChildAt(txt,1);
		stage.addChild(spr[sn]);
		spr[sn].scaleX = nume/deno;
		spr[sn].scaleY = nume/deno;
		spr[sn].visible = true;
		sn++;
		// ボタン関数
		btn_func[i] = new Function();
		btn_func[i] = null;
	}

	// スプライト枚数
	sn_max = sn;
	for( i=0; i<sn_max; i++ ) spr[i].visible = false;
	
	stage.addEventListener("stagemousedown", mouseDownListner );
	stage.addEventListener("stagemousemove", mouseMoveListner );
	stage.addEventListener("stagemouseup", mouseUpListner );
	createjs.Ticker.addEventListener("tick", onTick);
	createjs.Ticker.setFPS(60);
	
	if( soundon ){
		// PCの場合にはサウンド読み込む
		var queue = new createjs.LoadQueue(false);
		queue.installPlugin(createjs.Sound);
		queue.loadManifest(manifest,true);
		queue.addEventListener("fileload",handleFileLoad);  
		queue.addEventListener("complete",handleComplete);		
	}else{
		waitcount = 60;
		timer_func = fake_loading;
	}
}

function handleFileLoad(event){
	var item = event.item;
	if( item.type == createjs.LoadQueue.SOUND ){
		startSound(item.id);
	}	
}
function handleComplete(event){
	waitcount = 30;
	timer_func = fake_loading;
}
var instance = new Array();
function startSound(soundid){
	instance[soundid] = createjs.Sound.createInstance(soundid); // SoundJSのインスタンスを再生(idを指定)
}
function playSound(soundid){
	if( !soundon ) return;
	instance[soundid].setVolume(0.5);
	instance[soundid].play();
}

////////////////////////////////////////////////////
// イベントリスナー群
////////////////////////////////////////////////////

function mouseDownListner(e) {
	if( click_func != null ){ click_func(e); }
	canvas.style.cursor="default";  // マウスカーソルの変更
}
function mouseMoveListner(e) {
	if( move_func != null ){ move_func(e); }
	canvas.style.cursor="default";  // マウスカーソルの変更
}
function mouseUpListner(e) {
	if( release_func != null ){ release_func(e); }
	canvas.style.cursor="default";  // マウスカーソルの変更
	if( activebutton>=0 ){
		if( btn_func[activebutton] != null ){
			playSound("snd_button");
			btn_func[activebutton]();
		}
	}
}

function onTick() {
	if( timer_func != null ){ timer_func(); }
	check_button();
}

// ボタン
function check_button(){
	var i,sn;
	var n=-1;
	for( i=0; i<bmax; i++ ){
		sn = sn_btn+i;
		if( !spr[sn].visible ) continue;
		var pt = spr[sn].globalToLocal(stage.mouseX, stage.mouseY);
		if( spr[sn].hitTest(pt.x,pt.y) ) n=i;
	}
	if( activebutton == n ) return;
	activebutton = n;
	for( var i=0; i<bmax; i++ ){
		if( i==activebutton ){
			spr[sn_btn+i].getChildAt(0).gotoAndStop("press");
		}else{
			spr[sn_btn+i].getChildAt(0).gotoAndStop("btn");
		}
	}
	stage.update();
}

////////////////////////////////////////////////////
// Loading
////////////////////////////////////////////////////

function fake_loading(){
	spr[sn_load].visible = true;
	spr[sn_load].text = " ";
	spr[sn_mes].visible = true;
	spr[sn_mes].text = "Now loading... ";
	spr[sn_mes].x = view_w/2;
	spr[sn_mes].y = view_h/2;
	stage.update();
	waitcount--;
	if( waitcount<=0 ){
		timer_func = null;
		start_title();
	}
}

////////////////////////////////////////////////////
// タイトル画面
////////////////////////////////////////////////////

function start_title(){
	var i;
	
	for( i=0; i<sn_max; i++ ) spr[i].visible = false;
	
	spr[sn_title].visible = true;
	spr[sn_title].x = 0;
	spr[sn_title].y = 0;
	spr[sn_title].gotoAndStop("title");

	spr[sn_mes].visible = true;
	spr[sn_mes].text = "Copyright (C) 2001 GAMEDESIGN";
	spr[sn_mes].color = "#aaaaaa";
	spr[sn_mes].textAlign = "right";
	spr[sn_mes].x = view_w*0.9;
	spr[sn_mes].y = view_h*0.24;
	
	spr[sn_pmax].visible = true;
	for( i=0; i<7; i++ ){
		spr[sn_pmax].getChildByName("p"+i).color = (i==game.pmax-2)?"#aa0000":"#cccccc";
	}
	
	// ボタン
	spr[sn_btn+0].x = resize(640);
	spr[sn_btn+0].y = resize(390);
	spr[sn_btn+0].visible = true;
	btn_func[0] = make_map;
	spr[sn_btn+1].x = resize(640);
	spr[sn_btn+1].y = resize(490);
	spr[sn_btn+1].visible = true;
	btn_func[1] = toppage;

	stage.update();

	timer_func = null;
	click_func = click_pmax;
	move_func = null;
	releaese_func = null;	
}

function click_pmax(){
	var i,pmax=-1;
	for( i=0; i<7; i++ ){
		var o = spr[sn_pmax].getChildByName("p"+i);
		var pt = o.globalToLocal(stage.mouseX, stage.mouseY);
		if( Math.abs(pt.x)<(70*nume/deno) && Math.abs(pt.y)<(20*nume/deno) ){
			pmax = i+2;
		}
	}
	if( pmax<0 ) return;
	game.pmax = pmax;
	for( i=0; i<7; i++ ){
		spr[sn_pmax].getChildByName("p"+i).color = (i==game.pmax-2)?"#aa0000":"#cccccc";
	}
	stage.update();
}

////////////////////////////////////////////////////
// マップ作成画面
////////////////////////////////////////////////////

function make_map(){
	var i,j,x,y,n;
	
	for( i=0; i<sn_max; i++ ) spr[i].visible = false;

	game.make_map();
	
	// ダイスの表示順
	for( i=0; i<game.AREA_MAX; i++ ){
		n = prio[i].an;
		prio[i].cpos = game.adat[n].cpos;
	}
	for( i=0; i<game.AREA_MAX-1; i++ ){
		for( j=i; j<game.AREA_MAX; j++ ){
			if( prio[i].cpos>prio[j].cpos ){
				var tmp=prio[i].an; prio[i].an=prio[j].an; prio[j].an=tmp;
				tmp=prio[i].cpos; prio[i].cpos=prio[j].cpos; prio[j].cpos=tmp;
			}
		}
	}
	for( i=0; i<game.AREA_MAX; i++ ){
		n = prio[i].an;
		an2sn[n] = sn_dice+i;
	}

	// エリア塗り
	for( i=0; i<game.AREA_MAX; i++ ){
		draw_areashape(sn_area+i,i,0);
	}
	
	// エリアダイス
	for( i=0; i<game.AREA_MAX; i++ ){
		draw_areadice(sn_dice+i,prio[i].an);
	}

	spr[sn_mes].visible = true;
	spr[sn_mes].text = "Play this board?";
	spr[sn_mes].color = "#000000";
	spr[sn_mes].textAlign = "left";
	spr[sn_mes].x = view_w*0.1;
	spr[sn_mes].y = ypos_mes;

	// ボタン
	spr[sn_btn+2].x = resize(500);
	spr[sn_btn+2].y = ypos_mes;
	spr[sn_btn+2].visible = true;
	btn_func[2] = start_game;
	spr[sn_btn+3].x = resize(650);
	spr[sn_btn+3].y = ypos_mes;
	spr[sn_btn+3].visible = true;
	btn_func[3] = make_map;
	
	stage.update();	
	
	timer_func = null;
	click_func = null;
	move_func = null;
	releaese_func = null;	
}


function draw_areashape( sn, area, paint_mode ){
	var i,j;

	if( game.adat[area].size==0 ){
		spr[sn].visible = false;
		return;
	}
	spr[sn].visible = true;
	spr[sn].graphics.clear();
	var cnt = 0;
	var c = game.adat[area].line_cel[cnt];
	var d = game.adat[area].line_dir[cnt];
	var ax = [cel_w/2,cel_w,cel_w,cel_w/2,0,0,cel_w/2];
	var ax_left = [cel_w/2,cel_w,cel_w,cel_w/2,-cel_w/2,-cel_w/2,cel_w/2];
	var s = 3*nume/deno;
	var ay = [-s,s,cel_h-s,cel_h+s,cel_h-s,s,-s];
	var ay_top = [-cel_h/2,-cel_h/2,cel_h-s,cel_h+s,cel_h-s,-cel_h/2,-cel_h/2];
	var line_color = "#222244";
	if( paint_mode ) line_color = "#ff0000";
	spr[sn].graphics.beginStroke(line_color);
	var armcolor = ["#b37ffe","#b3ff01","#009302","#ff7ffe","#ff7f01","#b3fffe","#ffff01","#ff5858"];
	var color = armcolor[game.adat[area].arm];
	if( paint_mode ) color = "#000000";
	spr[sn].graphics.setStrokeStyle(4*nume/deno,"round","round").beginFill(color);
	var px=ax[d];
	var py=ay[d];
	spr[sn].graphics.moveTo( cpos_x[c]+px, cpos_y[c]+py );
	for( var i=0; i<100; i++ ){
		// まず線を引く
		var px=ax[d+1];
		var py=ay[d+1];
		spr[sn].graphics.lineTo(cpos_x[c]+px,cpos_y[c]+py);
		cnt++;
		c = game.adat[area].line_cel[cnt];
		d = game.adat[area].line_dir[cnt];
		if( c==game.adat[area].line_cel[0] && d==game.adat[area].line_dir[0] ) break;
	}
}

// エリアダイス
function draw_areadice(sn,area){
	if( game.adat[area].size==0 ){
		spr[sn].visible = false;
		return;
	}
	spr[sn].visible = true;
	var n = game.adat[area].cpos;
	spr[sn].x = Math.floor(cpos_x[n] + 6*nume/deno);
	spr[sn].y = Math.floor(cpos_y[n] - 10*nume/deno);
	spr[sn].gotoAndStop(game.adat[area].arm*10+game.adat[area].dice-1);
}

////////////////////////////////////////////////////
// プレイ開始
////////////////////////////////////////////////////

function start_game(){
	game.start_game();
	start_player();
}

// プレイヤー状態
function draw_player_data(){
	var i;
	var pnum = 0;
	for( i=0; i<8; i++ ){
		spr[sn_player+i].visible = false;
		var p = game.jun[i];
		if( game.player[p].area_tc > 0 ){
			spr[sn_player+i].visible = true;
			pnum++;
		}
	}
	var c=0;
	for( i=0; i<8; i++ ){
		var p = game.jun[i];
		if( game.player[p].area_tc == 0 )continue;
		var sn = sn_player+i;
		w = 100*nume/deno;
		var ox = view_w/2-(pnum-1)*w/2+c*w;
		spr[sn].x = ox;//-22*nume/deno;
		spr[sn].y = ypos_arm;
		spr[sn].getChildAt(0).gotoAndStop("d"+p+"0");
		spr[sn].getChildAt(1).text = ""+game.player[p].area_tc;
		spr[sn].getChildAt(2).text = "";
		if( game.player[p].stock>0 ) spr[sn].getChildAt(2).text = ""+game.player[p].stock;
		if( i==game.ban ){
			spr[sn_ban].x = ox;
			spr[sn_ban].y = ypos_arm;
			spr[sn_ban].gotoAndStop("ban");
			spr[sn_ban].visible = true;
		}
		c++;
	}
}

////////////////////////////////////////////////////
// 順番が来た
////////////////////////////////////////////////////

function start_player(){
	
	for( var i=sn_info; i<sn_max; i++ ){
		spr[i].visible = false;
	}
	draw_player_data();
	
	if( game.jun[game.ban] == game.user ){
		start_man();
	}else{
//		start_man();
		start_com();
	}
}

////////////////////////////////////////////////////
// プレイヤーの行動開始
////////////////////////////////////////////////////

function start_man(){
	
	spr[sn_mes].visible = true;
	spr[sn_mes].text = "1. Click your area. 2. Click neighbor to attack.";
	spr[sn_mes].color = "#000000";
	spr[sn_mes].textAlign = "left";
	spr[sn_mes].x = view_w*0.05;
	spr[sn_mes].y = ypos_mes;
	
	// ボタン
	activebutton = -1;	// ボタンをクリックしてないのにendturnになるバグ対応
	spr[sn_btn+4].x = view_w-100*nume/deno;
	spr[sn_btn+4].y = ypos_mes;
	spr[sn_btn+4].visible = true;
	btn_func[4] = end_turn;
	
	spr[sn_from].visible = false;
	spr[sn_to].visible = false;
	stage.update();
	
	timer_func = null;
	click_func = first_click;
	move_func = null;
	releaese_func = null;	
}

// クリックしたエリアの取得
function clicked_area(){
	var i,sn;
	var ret = -1;
	for( i=0; i<game.AREA_MAX; i++ ){
		if( game.adat[i].size==0 ) continue;
		sn = sn_area+i;
		var pt = spr[sn].globalToLocal(stage.mouseX, stage.mouseY);
		if( spr[sn].hitTest(pt.x,pt.y) ) ret=i;
	}
	for( i=0; i<game.AREA_MAX; i++ ){
		var a = prio[i].an;
		if( game.adat[a].size==0 ) continue;
		sn = sn_dice+i;
		var pt = spr[sn].globalToLocal(stage.mouseX, stage.mouseY);
		if( spr[sn].hitTest(pt.x,pt.y) ) ret=a;
	}
	return ret;
}

// 一回目
function first_click(){
	var p = game.jun[game.ban];
	var an = clicked_area();
	if( an<0 ) return;
	if( game.adat[an].arm != p ) return;
	if( game.adat[an].dice<=1 ) return;

	spr[sn_mes].visible = false;
		
	game.area_from = an;
	draw_areashape(sn_from,an,1);

	playSound("snd_click");

	stage.update();
	click_func = second_click;
}

// 二回目
function second_click(){
	var p = game.jun[game.ban];
	var an = clicked_area();
	if( an<0 ) return;
	
	// 同じエリアで選択解除	
	if( an==game.area_from ){
		start_man();
		return;
	}
	if( game.adat[an].arm == p ) return;
	if( game.adat[an].join[game.area_from]==0 ) return;
	
	game.area_to = an;
	draw_areashape(sn_to,an,1);
	stage.update();
	playSound("snd_click");
	start_battle();
}

// 行動終了
function end_turn(){

	spr[sn_btn+4].visible = false;
	spr[sn_from].visible = false;
	spr[sn_to].visible = false;
	spr[sn_mes].visible = false;
	
	timer_func = null;
	click_func = null;
	move_func = null;
	releaese_func = null;

	start_supply();
}

////////////////////////////////////////////////////
// COM思考
////////////////////////////////////////////////////

function start_com(){
	
	var ret = game.com_thinking();
	if( ret==0 ){
		start_supply();
		return;
	}
	stage.update();
	
	waitcount = 5;
	timer_func = com_from;
	click_func = null;
	move_func = null;
	releaese_func = null;
}

function com_from(){
	waitcount--;
	if( waitcount>0 ) return;
	
	draw_areashape(sn_from,game.area_from,1);
	stage.update();
	
	waitcount = 5;
	timer_func = com_to;
}

function com_to(){
	waitcount--;
	if( waitcount>0 ) return;

	draw_areashape(sn_to,game.area_to,1);
	stage.update();

	start_battle();
}


////////////////////////////////////////////////////
// 戦闘
////////////////////////////////////////////////////

function start_battle(){
	var i,j;
	
	spr[sn_btn+4].visible = false;	// END TURNボタン消す
	spr[sn_ban].visible = false;
	for( i=0; i<8; i++ ){
		spr[sn_player+i].visible = false;
	}

	// 戦闘シーンの変数	
	var an = [game.area_from,game.area_to];
	for( i=0; i<2; i++ ){
		battle[i].arm = game.adat[an[i]].arm;
		battle[i].dmax = game.adat[an[i]].dice;
		for( j=0; j<8; j++ ){
			var r = Math.floor(Math.random()*8);
			var tmp = battle[i].usedice[j];
			battle[i].usedice[j] = battle[i].usedice[r];
			battle[i].usedice[r] = tmp;
		}
		battle[i].sum=0;
		for( j=0; j<8; j++ ){
			battle[i].deme[j] = Math.floor(Math.random()*6);
			if( battle[i].usedice[j]<battle[i].dmax ){
				battle[i].sum += 1+battle[i].deme[j];
			}
			battle[i].fin[j] = false;
		}
	}
	spr[sn_battle].visible = true;
	
	for( i=0; i<2; i++ ){
		var w = 4;
		var h = 2;
		var r = 8;
		var ox = (i==0)?w*100:-w*90;
		var oy = (i==0)?-h*50:h*60;
		for( j=0; j<8; j++ ){
			var o = spr[sn_battle].getChildByName("d"+i+j);
			o.vx = ox + (j%3)*10*w - Math.floor(j/3)*10*w + Math.random()*r;
			o.vy = oy + (j%3)*10*h + Math.floor(j/3)*10*h + Math.random()*r;
			o.z = Math.random()*10;
			o.up = Math.random()*22;
			o.bc = 0;
			o.visible = false;
			var s = spr[sn_battle].getChildByName("s"+i+j);
			s.x = o.vx;
			s.y = o.vy;
			s.gotoAndStop("shadow");
			s.visible = false;
		}
	}
	spr[sn_battle].getChildByName("n0").x = 110;
	spr[sn_battle].getChildByName("n0").y = -10;
	spr[sn_battle].getChildByName("n0").visible = false;
	spr[sn_battle].getChildByName("n1").x = -290;
	spr[sn_battle].getChildByName("n1").y = -10;
	spr[sn_battle].getChildByName("n1").visible = false;
	
	bturn = 0;

	stage.update();
	timer_func = battle_dice;
	click_func = null;
	move_func = null;
	releaese_func = null;
}


function battle_dice(){
	var i,j;
	var w = (bturn==0)?-10:10;
	var h = (bturn==0)?6:-6;
	var f=false;
	var soundflg = false;
	for( i=0; i<8; i++ ){
		if( battle[bturn].fin[i]>0 ) continue;
		var o = spr[sn_battle].getChildByName("d"+bturn+i);
		o.visible = true;
		o.vx += w;
		o.vy += h;
		o.z += o.up;
		o.up -= 3;
		if( o.z<0 ){
			o.z = 0;
			o.up = 5-o.bc*3;
			o.bc++;
			if( o.bc>=2 ){
				battle[bturn].fin[i] = 1;
				if( bturn==0 ){
					if( i>=3 ){
						if( battle[bturn].fin[i-3]==0 ) battle[bturn].fin[i] = 0;
					}
					if( i>=2 ){
						if( battle[bturn].fin[i-2]==0 ) battle[bturn].fin[i] = 0;
					}
				}else{
					if( i<5 ){
						if( battle[bturn].fin[i+3]==0 ) battle[bturn].fin[i] = 0;
					}
					if( i<6 ){
						if( battle[bturn].fin[i+2]==0 ) battle[bturn].fin[i] = 0;
					}
				}
			}
			if( o.bc==1 ){
				if( battle[bturn].usedice[i]<battle[bturn].dmax ) soundflg = true;
			}
		}
		o.x = o.vx;
		o.y = o.vy-o.z;
		o.gotoAndStop("d"+battle[bturn].arm+Math.floor(Math.random()*6));
		if( battle[bturn].fin[i]>0 ){
			o.gotoAndStop("d"+battle[bturn].arm+battle[bturn].deme[i]);
			if( battle[bturn].usedice[i]<battle[bturn].dmax ) soundflg = true;
		}
		var s = spr[sn_battle].getChildByName("s"+bturn+i);
		s.visible = true;
		s.x = o.vx;
		s.y = o.vy;
		if( battle[bturn].usedice[i]>=battle[bturn].dmax ){
			o.visible = false;
			s.visible = false;
		}
		f=true;
	}
	if( !f ){
		spr[sn_battle].getChildByName("n"+bturn).visible = true;
		spr[sn_battle].getChildByName("n"+bturn).text = ""+battle[bturn].sum;
		bturn++;
		if( bturn>=2 ){
			waitcount=15;
			timer_func = after_battle;
		}
	}
	if( soundflg ){
		playSound("snd_dice");
	}
	stage.update();
}

function after_battle(){
	waitcount--;
	if( waitcount>0 ) return;
	spr[sn_battle].visible = false;
	spr[sn_from].visible = false;
	spr[sn_to].visible = false;
	spr[sn_ban].visible = true;
	for( i=0; i<8; i++ ){
		spr[sn_player+i].visible = true;
	}
	
	var arm0 = game.adat[game.area_from].arm;
	var arm1 = game.adat[game.area_to].arm;
	var defeat = ( battle[0].sum>battle[1].sum ) ? 1 : 0;
	if( defeat>0 ){
		game.adat[game.area_to].dice = game.adat[game.area_from].dice-1;
		game.adat[game.area_from].dice = 1;
		game.adat[game.area_to].arm = arm0;
		game.set_area_tc(arm0);
		game.set_area_tc(arm1);
		playSound("snd_success");
	}else{
		game.adat[game.area_from].dice = 1;
		playSound("snd_fail");
	}
	
	draw_areashape(sn_area+game.area_to,game.area_to,0);
	draw_areadice(an2sn[game.area_from],game.area_from);
	draw_areadice(an2sn[game.area_to],game.area_to);
	
	// 履歴
	game.set_his(game.area_from,game.area_to,defeat);

	// commented the code below to avoid gameover on player zero defeat
	// instead, play until one winner remains

	// if( game.player[game.user].area_tc==0 ){
	// 	draw_player_data();
	// 	start_gameover();
	// }else{
		var c=0;
		for( var i=0; i<game.pmax; i++ ){
			if( game.player[i].area_tc>0 ) c++;
		}
		if( c==1 ){
			draw_player_data();
			start_win();
		}else{
			start_player();
		}
	// }
}

////////////////////////////////////////////////////
// ダイス補充の開始
////////////////////////////////////////////////////

function start_supply(){
	spr[sn_from].visible = false;
	spr[sn_to].visible = false;
	spr[sn_btn+4].visible = false;

	var pn = game.jun[game.ban];
//	game.player[pn].stock = 64;
	game.set_area_tc(pn);
	game.player[pn].stock += game.player[pn].area_tc;
	if( game.player[pn].stock > game.STOCK_MAX ){
		game.player[pn].stock = game.STOCK_MAX;
	}
	
	spr[sn_supply].visible = true;
	for( var i=0; i<game.STOCK_MAX; i++ ){
		if( i<game.player[pn].stock ){
			spr[sn_supply].getChildAt(i).visible = true;
			spr[sn_supply].getChildAt(i).gotoAndStop("d"+pn+"3");
		}else{
			spr[sn_supply].getChildAt(i).visible = false;
		}
	}
	stage.addChild(spr[sn_supply]);
	
	waitcount = 10;
	timer_func = supply_waiting;
	click_func = null;
	move_func = null;
	releaese_func = null;	
}

function supply_waiting(){
	waitcount--;
	if( waitcount>0 ) return;
	timer_func = supply_dice;
}

function supply_dice(){
	var pn = game.jun[game.ban];
	var list = new Array();
	c = 0;
	for( var i=0; i<game.AREA_MAX; i++ ){
		if( game.adat[i].size == 0 ) continue;
		if( game.adat[i].arm != pn ) continue;
		if( game.adat[i].dice >= 8 ) continue;
		list[c] = i;
		c++;
	}
	if( c==0 || game.player[pn].stock<=0 ){
		next_player();
		return;
	}
	
	game.player[pn].stock--;
	var an = list[Math.floor(Math.random()*c)];
	game.adat[an].dice++;
	draw_areadice(an2sn[an],an);
	
	for( i=0; i<game.STOCK_MAX; i++ ){
		if( i<game.player[pn].stock ){
			spr[sn_supply].getChildAt(i).visible = true;
		}else{
			spr[sn_supply].getChildAt(i).visible = false;
		}
	}
	// 履歴
	game.set_his(an,0,0);

	stage.update();
	
	return;
}


////////////////////////////////////////////////////
// 次のプレイヤーへ
////////////////////////////////////////////////////

function next_player(){
	for( var i=0; i<game.pmax; i++ ){
		game.ban++;
		if( game.ban >= game.pmax ) game.ban = 0;
		var pn = game.jun[game.ban];
		if( game.player[pn].area_tc ) break;
	}
	if( game.jun[game.ban] == game.user ) playSound("snd_myturn");

	start_player();
}

////////////////////////////////////////////////////
// GAMEOVER
////////////////////////////////////////////////////

function start_gameover(){
	spr[sn_gameover].visible = false;
	spr[sn_gameover].x = view_w/2;
	spr[sn_gameover].y = view_h/2;
	spr[sn_gameover].getChildByName("bg").alpha = 0;
	spr[sn_gameover].getChildByName("mes").alpha = 0;
	spr[sn_gameover].getChildByName("mes").y = -120;
	stage.update();
	stat = 0;
	waitcount = 0;
	timer_func = gameover;
	click_func = null;
	move_func = null;
	releaese_func = null;	
}

function gameover(){
	
	spr[sn_gameover].visible = true;
	waitcount++;
	if( stat==0 ){
		var a = (-80+waitcount)/100;
		spr[sn_gameover].getChildByName("bg").alpha=a;
		if( a>0.8 ){
			playSound("snd_over");
			waitcount=0;
			stat++;
		}
		stage.update();
	}else if( stat==1 ){
		var a = waitcount/100;
		var o = spr[sn_gameover].getChildByName("mes");
		o.alpha=a;
		o.y+=0.5;
		if( o.y>-70 ) o.y=-70;
		if( waitcount>=160 ){
			// ボタン
			spr[sn_btn+5].x = view_w/2 - resize(100);
			spr[sn_btn+5].y = view_h/2 + resize(70);
			spr[sn_btn+5].visible = true;
			btn_func[5] = start_title;
			spr[sn_btn+6].x = view_w/2 + resize(100);
			spr[sn_btn+6].y = view_h/2 + resize(70);
			spr[sn_btn+6].visible = true;
			btn_func[6] = start_history;
			
			waitcount=0;
			stat++;
		}
		stage.update();
	}
}

////////////////////////////////////////////////////
// YOU WIN!
////////////////////////////////////////////////////

function start_win(){
	spr[sn_win].visible = false;
	spr[sn_win].x = view_w/2;
	spr[sn_win].y = view_h/2 - resize(70);
	spr[sn_win].gotoAndStop("win");
	waitcount = 0;
	timer_func = win;
	click_func = null;
	move_func = null;
	releaese_func = null;	
}

function win(){
	waitcount++;
	var a = Math.floor(waitcount/2);
	if( a==10 || a==12 || a==14 || a==16 || a>=18 ){
		spr[sn_win].visible = true;
	}else{
		spr[sn_win].visible = false;
	}
	if( a==10 ){
		playSound("snd_clear");
	}
	
	if( a>=40 ){
		timer_func = null;
		spr[sn_btn+6].x = view_w/2;
		spr[sn_btn+6].y = view_h/2 + resize(70);
		spr[sn_btn+6].visible = true;
		btn_func[6] = start_history;
	}
	stage.update();
}

////////////////////////////////////////////////////
// 履歴
////////////////////////////////////////////////////

function start_history(){
	var i;
	
	spr[sn_win].visible = false;
	spr[sn_gameover].visible = false;
	spr[sn_ban].visible = false;
	for( i=0; i<8; i++ ) spr[sn_player+i].visible = false;
	for( i=0; i<bmax; i++ ) spr[sn_btn+i].visible = false;

	for( i=0; i<game.AREA_MAX; i++ ){
		if( game.adat[i].size==0 ) continue;
		game.adat[i].dice = game.his_dice[i];
		game.adat[i].arm = game.his_arm[i];
		draw_areashape(sn_area+i,i,0);
	}
	for( i=0; i<game.AREA_MAX; i++ ){
		draw_areadice(sn_dice+i,prio[i].an);
	}
	
	// ボタン
	spr[sn_btn+5].x = view_w/2 - resize(100);
	spr[sn_btn+5].y = view_h*0.88;
	spr[sn_btn+5].visible = true;
	btn_func[5] = start_title;
	spr[sn_btn+1].x = view_w/2 + resize(100);
	spr[sn_btn+1].y = view_h*0.88;
	spr[sn_btn+1].visible = true;
	btn_func[1] = toppage;
	
	stage.addChild(spr[sn_btn+5]);
	stage.addChild(spr[sn_btn+1]);
	stage.update();
	replay_c = 0;
	stat = 0;
	waitcount = 0;
	timer_func = play_history;
	click_func = null;
	move_func = null;
	releaese_func = null;	
}

function play_history(){

	var an;
	if( stat==0 ){
		if( replay_c >= game.his_c ){
			timer_func = null;	// 終了
		}else{
			stat = ( game.his[replay_c].to==0 ) ? 1 : 2;
		}
	}else if( stat==1 ){
		// 補給
		an = game.his[replay_c].from;
		game.adat[an].dice++;
		draw_areadice(an2sn[an],an);
		stage.update();
		replay_c++;
		if( replay_c >= game.his_c ){
			timer_func = null;	// 終了
		}else{
			stat = ( game.his[replay_c].to==0 ) ? 1 : 2;
		}
	}else if( stat==2 ){
		// 攻撃元
		an = game.his[replay_c].from;
		draw_areashape(sn_from,an,1);
		stage.update();
		waitcount=0;
		stat++;
	}else if( stat==3 ){
		// 攻撃先
		if( waitcount>2 ){
			an = game.his[replay_c].to;
			draw_areashape(sn_to,an,1);
			stage.update();
			waitcount=0;
			stat++;
		}
	}else if( stat==4 ){
		// 攻撃後
		if( waitcount>10 ){
			var an0 = game.his[replay_c].from;
			var an1 = game.his[replay_c].to;
			if( game.his[replay_c].res>0 ){
				game.adat[an1].dice = game.adat[an0].dice-1;
				game.adat[an0].dice = 1;
				game.adat[an1].arm = game.adat[an0].arm;
				playSound("snd_success");
			}else{
				game.adat[an0].dice = 1;
				playSound("snd_fail");
			}
			spr[sn_from].visible = false;
			spr[sn_to].visible = false;
			draw_areadice(an2sn[an0],an0);
			draw_areadice(an2sn[an1],an1);
			draw_areashape(sn_area+an1,an1,0);
			stage.update();
			stat=0;
			replay_c++;
		}
	}
	waitcount++;
}

////////////////////////////////////////////////////
// リンク
////////////////////////////////////////////////////

function toppage(){
	location.href="https://www.gamedesign.jp/";
}

////////////////////////////////////////////////////
// Programmatic AI Interface
////////////////////////////////////////////////////

// Data extraction
function get_area_data() {
	let internal_area_ids = [];
	let area_id_to_index = [];
	let node_positions = [];

	// Collect existing areas
	for(i = 0; i < game.AREA_MAX; i++){
		if(game.adat[i].size > 0){
			area_id_to_index.push(internal_area_ids.length);
			internal_area_ids.push(i);
		} else {
			area_id_to_index.push(-1);
		}
	}
	
	let num_areas = internal_area_ids.length;

	let adjacency = new Array(num_areas);
	for (let i = 0; i < num_areas; i++){
		adjacency[i] = new Array(num_areas).fill(0);
	}

	for (let i = 0; i < num_areas; i++){
		let area_id = internal_area_ids[i];
		let area_data = game.adat[area_id];

		node_positions.push([area_data.cx, area_data.cy]);
		// Set adjacency relationships
		for (let j = 0; j < num_areas; j++){
			let other_area_id = internal_area_ids[j];
			if(area_data.join[other_area_id] > 0){
				adjacency[i][j] = 1;
			}
		}
	}

	return {
		adjacency: adjacency, 
		internal_area_ids: internal_area_ids,
		node_positions: node_positions,
		area_id_to_index: area_id_to_index
	};
}

function get_curret_game_state(){
	let result = {
		teams: [],
		dice: []
	};
	
	// Fill data for existing areas only
	for (let i = 0; i < game.AREA_MAX; i++){
		if(game.adat[i].size <= 0) {
			continue;
		}
		// Set team and dice count
		result.teams.push(game.adat[i].arm);
		result.dice.push(game.adat[i].dice);
	}

	return result;
}

// Make AI move for current player
function make_ai_move(){
	start_com();
	
	let move_made = (timer_func === com_from);
	let turn_end = (timer_func === supply_waiting);

	return {
		success: true,
		from: move_made ? game.area_from : null,
		to: move_made ? game.area_to : null,
		move_made: move_made,
		turn_end: turn_end
	};
}

// Simulate a move (for external control)
function simulate_move(from_area_id, to_area_id){
	// Validate the move
	if (click_func != first_click) {
		return {success: false, error: "Game does not expect a move right now"};
	}
	if(game.adat[from_area_id].size == 0 || game.adat[to_area_id].size == 0){
		return {success: false, error: "Invalid areas"};
	}
	if(game.adat[from_area_id].arm != game.get_pn()){
		return {success: false, error: "Not your area"};
	}
	if(game.adat[from_area_id].join[to_area_id] == 0){
		return {success: false, error: "Areas not adjacent"};
	}
	if(game.adat[from_area_id].dice <= 1){
		return {success: false, error: "Need at least 2 dice to attack"};
	}
	
	// Set the attack parameters
	game.area_from = from_area_id;
	game.area_to = to_area_id;
	
	// Execute the battle directly
	// start_battle();
	
	// start the "timer" that would visualise the move 
	// and call start_battle() internally
	waitcount = 5;
	timer_func = com_from;
	click_func = null;
	move_func = null;
	releaese_func = null;

	return {
		success: true,
		from: from_area_id,
		to: to_area_id
	};
}

// End turn programmatically
function end_turn_programmatic(){
	if (click_func != first_click) {
		return {success: false, error: "Game does not expect a move right now"};
	}
	// Call the existing end_turn function
	end_turn();
	
	return {
		success: true,
		current_player: game.get_pn(),
		turn_index: game.ban
	};
}

function manual_only() {
	start_com = start_man
}

function ai_only() {
	start_man = start_com
}

let game_history = {
	internal_area_ids: [],
	area_id_to_index: [],
	node_positions: [],
	adjacency: [],
	states: [],
	actions: []
}

function get_game_state() {
    const last_state = game_history.states.length > 0 
        ? game_history.states[game_history.states.length - 1] 
        : null;
    
    return {
        internal_area_ids: game_history.internal_area_ids,
        area_id_to_index: game_history.area_id_to_index,
        node_positions: game_history.node_positions,
        adjacency: game_history.adjacency,
        state: last_state
    };
}

async function save_history() {
	try {
		// Send to server via API call
		console.log('Sending game history to server...');
		const response = await fetch('/api/save-history', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify(game_history)
		});
		
		const data = await response.json();
		console.log('Game history saved to server:', data);
		return data;
	} catch (error) {
		console.error('Error saving game history:', error);
	}
}

let history_saved_wrap = false;
let last_was_attack = false;
let last_was_endturn = false;

function get_last_action_data() {
	let area_from = game_history.area_id_to_index[game.area_from];
	let area_to = game_history.area_id_to_index[game.area_to];
	
	game_history.internal_area_ids

	return {
		player: game.get_pn(),
		from: last_was_attack ? area_from : null,
		to: last_was_attack ? area_to : null,
		move_made: last_was_attack,
		turn_end: last_was_endturn
	};
}

function save_area_data() {
	let area_data = get_area_data();
	game_history.internal_area_ids = area_data.internal_area_ids;
	game_history.area_id_to_index = area_data.area_id_to_index;
	game_history.adjacency = area_data.adjacency;
	game_history.node_positions = area_data.node_positions;
}

function extend_history() {
	let game_state = get_curret_game_state();
	let action_data = get_last_action_data();
	game_history.states.push(game_state);
	game_history.actions.push(action_data);
}

function maintain_game_history() {
	if (history_saved_wrap) { return false; }
	history_saved_wrap = true;

	let old_start_game = start_game;
	let new_start_game = function() {
		save_area_data();
		old_start_game();
		console.log("area ids saved")
	};
	start_game = new_start_game;

	let old_start_battle = start_battle;
	let new_start_battle = function() {
		last_was_attack = true;
		last_was_endturn = false;
		// history is extended before each battle and each end-turn-resupply
		extend_history();
		console.log("new_start_battle history saved")
		old_start_battle();
	};
	start_battle = new_start_battle;

	let old_start_supply = start_supply;
	let new_start_supply = function() {
		last_was_attack = false;
		last_was_endturn = true;
		// history is extended before each battle and each end-turn-resupply
		extend_history();
		console.log("new_start_supply history saved")
		old_start_supply();
	};
	start_supply = new_start_supply;

	let old_start_win = start_win;
	let new_start_win = function() {
		// save the terminal state, dont save action - there was none
		game_history.states.push(get_curret_game_state());
		old_start_win();
		save_history();
	};
	start_win = new_start_win;

	let old_start_gameover = start_gameover;
	let new_start_gameover = function() {
		// save the terminal state, dont save action - there was none
		game_history.states.push(get_curret_game_state());
		old_start_gameover();
		save_history();
	};
	start_gameover = new_start_gameover;
}

function restart() {
	Object.keys(game_history).forEach(key => {
		game_history[key] = [];
	});
	console.log("history cleaned")

	make_map();
	start_game();
}

function auto_restart() {
	let old_start_win = start_win;
	let new_start_win = function() {
		old_start_win();
		restart();
	};
	start_win = new_start_win;

	let old_start_gameover = start_gameover;
	let new_start_gameover = function() {
		old_start_gameover();
		restart();
	};
	start_gameover = new_start_gameover;
}

function autoplay() {
	maintain_game_history();
	auto_restart();
	ai_only();
}

function faster() {
	createjs.Ticker.setFPS(600);
}