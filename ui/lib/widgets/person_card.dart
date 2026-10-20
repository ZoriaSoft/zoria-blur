import 'dart:io';

import 'package:flutter/material.dart';

import '../models.dart';
import '../strings.dart';
import '../theme.dart';

/// One person in the inspector grid: face, name, how long they are on screen. Click = select.
class PersonCard extends StatefulWidget {
  const PersonCard({
    super.key,
    required this.person,
    required this.selected,
    required this.willBlur,
    required this.onTap,
    required this.onRename,
    this.onShow,
  });
  final Person person;
  final bool selected;
  final bool willBlur; // with the current mode: will this face be blurred in the export
  final VoidCallback onTap;
  final VoidCallback onRename;
  final VoidCallback? onShow;

  @override
  State<PersonCard> createState() => _PersonCardState();
}

class _PersonCardState extends State<PersonCard> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    final c = context.pal;
    final p = widget.person;
    final sel = widget.selected;
    Widget noFace() => ColoredBox(color: c.hover, child: Icon(Icons.person_outline, size: 32, color: c.faint));
    return MouseRegion(
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: widget.onTap,
        child: Container(
          decoration: BoxDecoration(
            color: sel ? c.accentSoft : (_hover ? c.hover : c.surface),
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: sel ? c.accent : c.border, width: sel ? 1.5 : 1),
          ),
          padding: EdgeInsets.all(sel ? 3.5 : 4),
          child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Expanded(
              child: Stack(fit: StackFit.expand, children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(2),
                  child: p.thumbnails.isEmpty
                      ? noFace()
                      : Image.file(
                          File(p.thumbnails.first),
                          fit: BoxFit.cover,
                          cacheWidth: 256,
                          filterQuality: FilterQuality.medium,
                          errorBuilder: (_, _, _) => noFace(),
                        ),
                ),
                Positioned(
                  left: 4,
                  top: 4,
                  child: Container(
                    width: 18,
                    height: 18,
                    decoration: BoxDecoration(
                      color: sel ? c.accent : Colors.white.withValues(alpha: 0.92),
                      borderRadius: BorderRadius.circular(3),
                      border: Border.all(color: sel ? c.accent : const Color(0xFF9AA1AB)),
                    ),
                    child: sel ? Icon(Icons.check, size: 14, color: c.onAccent) : null,
                  ),
                ),
                if (widget.willBlur)
                  Positioned(
                    left: 4,
                    bottom: 4,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                      decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.66), borderRadius: BorderRadius.circular(3)),
                      child: const Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(Icons.blur_on, size: 12, color: Colors.white),
                        SizedBox(width: 3),
                        Text(S.willBlur,
                            style: TextStyle(color: Colors.white, fontSize: 10.5, fontWeight: FontWeight.w600)),
                      ]),
                    ),
                  ),
                if (_hover && widget.onShow != null)
                  Positioned(
                    right: 4,
                    bottom: 4,
                    child: Tooltip(
                      message: S.preview,
                      child: Material(
                        color: Colors.black.withValues(alpha: 0.66),
                        borderRadius: BorderRadius.circular(3),
                        child: InkWell(
                          borderRadius: BorderRadius.circular(3),
                          onTap: widget.onShow,
                          child: const Padding(
                            padding: EdgeInsets.all(3),
                            child: Icon(Icons.visibility_outlined, size: 14, color: Colors.white),
                          ),
                        ),
                      ),
                    ),
                  ),
              ]),
            ),
            const SizedBox(height: Sp.xs),
            Row(children: [
              Expanded(
                child: Text(p.displayName,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600, color: c.text)),
              ),
              Tooltip(
                message: S.rename,
                child: InkWell(
                  borderRadius: BorderRadius.circular(3),
                  onTap: widget.onRename,
                  child: Padding(
                    padding: const EdgeInsets.all(2),
                    child: Icon(Icons.edit_outlined, size: 14, color: c.muted),
                  ),
                ),
              ),
            ]),
            Text(S.seconds(fmtSeconds(p.seconds)),
                maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 11.5, color: c.muted)),
          ]),
        ),
      ),
    );
  }
}
