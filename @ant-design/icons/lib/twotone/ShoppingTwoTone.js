"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ShoppingTwoTone = {
    name: 'shopping',
    theme: 'twotone',
    icon: function (primaryColor, secondaryColor) {
        return {
            tag: 'svg',
            attrs: { viewBox: '64 64 896 896' },
            children: [
                {
                    tag: 'path',
                    attrs: {
                        fill: secondaryColor,
                        d: 'M696 472c0 4.4-3.6 8-8 8h-56c-4.4 0-8-3.6-8-8v-88H400v88c0 4.4-3.6 8-8 8h-56c-4.4 0-8-3.6-8-8v-88h-96v456h560V384h-96v88z'
                    }
                },
                {
                    tag: 'path',
                    attrs: {
                        d: 'M832 312H696v-16c0-101.6-82.4-184-184-184s-184 82.4-184 184v16H192c-17.7 0-32 14.3-32 32v536c0 17.7 14.3 32 32 32h640c17.7 0 32-14.3 32-32V344c0-17.7-14.3-32-32-32zm-432-16c0-61.9 50.1-112 112-112s112 50.1 112 112v16H400v-16zm392 544H232V384h96v88c0 4.4 3.6 8 8 8h56c4.4 0 8-3.6 8-8v-88h224v88c0 4.4 3.6 8 8 8h56c4.4 0 8-3.6 8-8v-88h96v456z',
                        fill: primaryColor
                    }
                }
            ]
        };
    }
};
exports.default = ShoppingTwoTone;
