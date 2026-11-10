/*
 * MPRSS Module 1 - Behavioral Hook Agent (Frida), v2
 *
 * Reports two kinds of events, both as JSON via send():
 *   { ts, category, api, detail, pid }
 *
 *  1. MODEL events: calls to the 44 APIs in vocab_v1.txt. `api` is EXACTLY the name used in the
 *     training data ('package.Class.method'), because the Module 1 service matches on it.
 *  2. ALERT events: rare but critical APIs (SMS, device IDs, location, camera, content queries,
 *     dynamic dex loading). The model never learned these (almost absent from the training
 *     traces); they are shown on the dashboard as raw alerts.
 *
 * Matching the training traces (which only record calls made BY THE APP'S OWN CODE):
 *  - CALLER_FILTER: a model event is reported only if the immediate caller is not an Android/Java
 *    framework class. Android calls many of these APIs itself, and counting those would not match
 *    what the model was trained on.
 *  - Every overload of every method is hooked (the training names ignore parameter types).
 *  - Abstract classes cannot be hooked directly; HOOK_ON maps them to the classes that really run.
 *  - After MAX_VOCAB_EVENTS accepted events the agent stops inspecting calls (the model only reads
 *    the first 64), so the overhead on the app stays bounded.
 *
 * Passive observation only: every hook calls through to the real implementation.
 * Send {type:'hook_status', ...} once after installation so problems are visible, not silent.
 */

import Java from 'frida-java-bridge';

// ---------------------------------------------------------------- settings
var CALLER_FILTER = true;      // set to false to see ALL calls (debugging: if no events appear, try this)
var DEBUG_STACK = false;       // true: print the first few call stacks to the Frida console
var MAX_VOCAB_EVENTS = 300;
var MAX_ALERT_EVENTS = 200;

// ---------------------------------------------------------------- model vocabulary (must equal vocab_v1.txt)
var VOCAB_TOKENS = [
    ['android.telephony.TelephonyManager.getNetworkOperator', 'telephony'],
    ['android.telephony.TelephonyManager.getSimOperator', 'telephony'],
    ['android.telephony.TelephonyManager.getSimCountryIso', 'telephony'],
    ['android.telephony.TelephonyManager.getNetworkCountryIso', 'telephony'],
    ['android.telephony.TelephonyManager.listen', 'telephony'],
    ['android.net.ConnectivityManager.getActiveNetworkInfo', 'network'],
    ['android.net.ConnectivityManager.getNetworkInfo', 'network'],
    ['java.net.URL.<init>', 'network'],
    ['java.net.URL.getHost', 'network'],
    ['java.net.URL.openConnection', 'network'],
    ['java.net.URLConnection.setRequestProperty', 'network'],
    ['java.net.HttpURLConnection.setRequestMethod', 'network'],
    ['java.lang.Runtime.exec', 'command_exec'],
    ['java.lang.reflect.AccessibleObject.setAccessible', 'reflection'],
    ['java.lang.reflect.Field.get', 'reflection'],
    ['java.lang.Class.forName', 'reflection'],
    ['java.lang.Class.getMethod', 'reflection'],
    ['javax.crypto.Cipher.getInstance', 'crypto'],
    ['javax.crypto.Cipher.doFinal', 'crypto'],
    ['javax.crypto.spec.SecretKeySpec.<init>', 'crypto'],
    ['java.security.MessageDigest.getInstance', 'crypto'],
    ['android.util.Base64.decode', 'crypto'],
    ['android.content.pm.PackageManager.getPackageInfo', 'package_info'],
    ['android.content.pm.PackageManager.getApplicationInfo', 'package_info'],
    ['android.content.pm.PackageManager.hasSystemFeature', 'package_info'],
    ['android.content.pm.PackageManager.getActivityInfo', 'package_info'],
    ['android.content.pm.PackageManager.checkPermission', 'package_info'],
    ['android.content.pm.PackageManager.resolveActivity', 'package_info'],
    ['android.content.pm.PackageManager.getInstallerPackageName', 'package_info'],
    ['android.provider.Settings$Secure.getString', 'package_info'],
    ['java.io.File.exists', 'file_io'],
    ['java.io.File.mkdir', 'file_io'],
    ['java.io.File.delete', 'file_io'],
    ['java.io.File.getParentFile', 'file_io'],
    ['java.io.FileOutputStream.<init>', 'file_io'],
    ['java.io.FileInputStream.<init>', 'file_io'],
    ['android.os.Environment.getExternalStorageDirectory', 'file_io'],
    ['android.content.ContextWrapper.registerReceiver', 'component'],
    ['android.content.ContextWrapper.bindService', 'component'],
    ['android.content.ContextWrapper.startService', 'component'],
    ['android.webkit.WebView.loadUrl', 'webview'],
    ['android.webkit.WebSettings.setJavaScriptEnabled', 'webview'],
    ['android.webkit.WebSettings.setAllowFileAccess', 'webview'],
    ['android.webkit.WebView.setWebViewClient', 'webview']
];

// ---------------------------------------------------------------- raw alerts (NOT part of the model vocabulary)
var ALERT_TOKENS = [
    ['android.telephony.SmsManager.sendTextMessage', 'sms'],
    ['android.telephony.SmsManager.sendMultipartTextMessage', 'sms'],
    ['android.telephony.TelephonyManager.getDeviceId', 'telephony'],
    ['android.telephony.TelephonyManager.getSubscriberId', 'telephony'],
    ['android.telephony.TelephonyManager.getSimSerialNumber', 'telephony'],
    ['android.telephony.TelephonyManager.getLine1Number', 'telephony'],
    ['android.location.LocationManager.getLastKnownLocation', 'location'],
    ['android.hardware.Camera.open', 'camera'],
    ['android.hardware.camera2.CameraManager.openCamera', 'camera'],
    ['android.content.ContentResolver.query', 'content_query'],
    ['dalvik.system.DexClassLoader.<init>', 'dynamic_loading']
];

// ---------------------------------------------------------------- which classes really run the code
var OKHTTP = ['com.android.okhttp.internal.huc.HttpURLConnectionImpl',
              'com.android.okhttp.internal.huc.HttpsURLConnectionImpl'];
var HOOK_ON = {
    // PackageManager is abstract: the app talks to ApplicationPackageManager
    'android.content.pm.PackageManager': {
        hook: ['android.app.ApplicationPackageManager'],
        skip: ['android.content.pm.PackageManager', 'android.app.ApplicationPackageManager']
    },
    // URL.openConnection() really returns okhttp's implementation classes on Android
    'java.net.URLConnection': {
        hook: ['java.net.URLConnection'].concat(OKHTTP),
        skip: ['java.net.URLConnection', 'java.net.HttpURLConnection'].concat(OKHTTP)
    },
    'java.net.HttpURLConnection': {
        hook: ['java.net.HttpURLConnection'].concat(OKHTTP),
        skip: ['java.net.URLConnection', 'java.net.HttpURLConnection'].concat(OKHTTP)
    }
};

// ---------------------------------------------------------------- caller classification
var FRAMEWORK_PREFIXES = ['java.', 'javax.', 'android.', 'dalvik.', 'com.android.', 'libcore.', 'sun.',
                          'org.apache.harmony.', 'org.json.', 'org.xmlpull.', 'org.w3c.', 'org.xml.'];
// libraries bundled INSIDE apps; the training traces count their calls as app calls
var APP_SIDE_PREFIXES = ['android.support.', 'android.arch.', 'android.databinding.'];

function startsWithAny(name, prefixes) {
    for (var i = 0; i < prefixes.length; i++) {
        if (name.indexOf(prefixes[i]) === 0) return true;
    }
    return false;
}

function isFrameworkClass(name) {
    if (startsWithAny(name, APP_SIDE_PREFIXES)) return false;
    return startsWithAny(name, FRAMEWORK_PREFIXES);
}

// ---------------------------------------------------------------- helpers
function str(x) {
    try {
        if (x === null || x === undefined) return '';
        var s = String(x);
        return s.length > 150 ? s.substring(0, 150) : s;
    } catch (e) {
        return '';
    }
}

function report(category, api, detail) {
    send({ ts: Date.now(), category: category, api: api, detail: detail || '', pid: Process.id });
}

// short, safe details for a few APIs (never device identifiers)
var DETAIL = {
    'java.lang.Runtime.exec': function (t, a) { return str(a[0]); },
    'java.net.URL.openConnection': function (t, a) { return str(t); },
    'java.net.URL.<init>': function (t, a) { return typeof a[0] === 'string' ? str(a[0]) : ''; },
    'java.io.FileOutputStream.<init>': function (t, a) { return str(a[0]); },
    'java.io.FileInputStream.<init>': function (t, a) { return str(a[0]); },
    'android.webkit.WebView.loadUrl': function (t, a) { return str(a[0]); },
    'android.provider.Settings$Secure.getString': function (t, a) { return str(a[1]); },
    'android.telephony.SmsManager.sendTextMessage': function (t, a) { return 'to=' + str(a[0]); },
    'android.location.LocationManager.getLastKnownLocation': function (t, a) { return 'provider=' + str(a[0]); },
    'android.content.ContentResolver.query': function (t, a) {
        var u = str(a[0]);
        var kind = 'content';
        if (u.indexOf('contacts') !== -1) kind = 'contacts';
        else if (u.indexOf('sms') !== -1) kind = 'sms';
        else if (u.indexOf('call_log') !== -1 || u.indexOf('calls') !== -1) kind = 'call_log';
        return kind + ':' + u;
    }
};

var vocabCount = 0;
var alertCount = 0;
var stackDebugShown = 0;
var Exception = null;      // java.lang.Exception, set once Java is available

// The immediate caller of the hooked method: the first stack frame that is not the hooked method itself.
function immediateCaller(skipClasses, methodName) {
    var st = Exception.$new().getStackTrace();
    var n = st.length;
    var h = -1;
    for (var i = 0; i < n && i < 12; i++) {
        if (skipClasses.indexOf(st[i].getClassName()) !== -1 && st[i].getMethodName() === methodName) { h = i; break; }
    }
    var j = h === -1 ? 0 : h;                      // hooked frame not in the trace: frame 0 is already the caller
    while (j < n && skipClasses.indexOf(st[j].getClassName()) !== -1 && st[j].getMethodName() === methodName) j++;
    if (DEBUG_STACK && stackDebugShown < 5) {
        stackDebugShown++;
        var lines = [];
        for (var k = 0; k < n && k < 6; k++) lines.push('    ' + st[k].getClassName() + '.' + st[k].getMethodName());
        console.log('[stack for ' + skipClasses[0] + '.' + methodName + '] hooked frame index=' + h + '\n' + lines.join('\n'));
    }
    return j < n ? st[j].getClassName() : null;
}

function onCall(spec, thisObj, args) {
    if (spec.alert) {
        if (alertCount >= MAX_ALERT_EVENTS) return;
        alertCount++;
        var ad = DETAIL[spec.token];
        report(spec.category, spec.token, ad ? ad(thisObj, args) : '');
        return;
    }
    if (vocabCount >= MAX_VOCAB_EVENTS) return;            // cap reached: no more inspection, no overhead
    if (CALLER_FILTER) {
        var caller = immediateCaller(spec.skip, spec.method);
        if (caller === null || isFrameworkClass(caller)) return;
    }
    vocabCount++;
    var d = DETAIL[spec.token];
    report(spec.category, spec.token, d ? d(thisObj, args) : '');
}

// Hook every overload of one method of one class. Returns how many overloads were hooked.
function hookMethod(className, methodName, spec) {
    var C = Java.use(className);
    var key = methodName === '<init>' ? '$init' : methodName;
    var m = C[key];
    if (m === undefined || m === null) throw new Error('method not found');
    var overloads = m.overloads;
    overloads.forEach(function (ov) {
        ov.implementation = function () {
            try { onCall(spec, this, arguments); } catch (e) { /* never break the app */ }
            return ov.apply(this, arguments);
        };
    });
    return overloads.length;
}

function install(token, category, isAlert, status) {
    var dot = token.lastIndexOf('.');
    var cls = token.substring(0, dot);
    var method = token.substring(dot + 1);
    var plan = HOOK_ON[cls] || { hook: [cls], skip: [cls] };
    var spec = { token: token, category: category, alert: isAlert, method: method, skip: plan.skip };
    var hooked = 0;
    var reasons = [];
    plan.hook.forEach(function (hc) {
        try { hooked += hookMethod(hc, method, spec); }
        catch (e) { reasons.push(hc + ': ' + e); }
    });
    if (hooked > 0) {
        (isAlert ? status.alerts_installed : status.installed).push(token);
        if (reasons.length > 0) status.partial.push({ token: token, reasons: reasons });
    } else {
        status.failed.push({ token: token, reason: reasons.join(' | ') });
    }
}

// ---------------------------------------------------------------- startup
var waitAttempts = 0;
function waitForJava(callback) {
    waitAttempts++;
    if (waitAttempts % 20 === 0) console.log('[behavior_hooks] Still waiting for Java... attempt ' + waitAttempts);
    if (typeof Java !== 'undefined' && Java.available) {
        callback();
    } else {
        setTimeout(function () { waitForJava(callback); }, 100);
    }
}

waitForJava(function () {
    try {
        Java.perform(function () {
            Exception = Java.use('java.lang.Exception');
            var status = { type: 'hook_status', installed: [], alerts_installed: [], failed: [], partial: [],
                           caller_filter: CALLER_FILTER };
            VOCAB_TOKENS.forEach(function (t) { install(t[0], t[1], false, status); });
            ALERT_TOKENS.forEach(function (t) { install(t[0], t[1], true, status); });
            console.log('[behavior_hooks] model APIs hooked: ' + status.installed.length + '/' + VOCAB_TOKENS.length +
                        ', alert APIs: ' + status.alerts_installed.length + '/' + ALERT_TOKENS.length +
                        ', failed: ' + status.failed.length);
            send(status);
        });
    } catch (outerErr) {
        console.log('[behavior_hooks] FATAL: ' + outerErr + '\n' + outerErr.stack);
    }
});