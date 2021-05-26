using DwgDump.Data;
using Multicad;
using Multicad.DatabaseServices;
using Multicad.Runtime;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace DwgDump
{
	public static class Main
	{
		static CrawlAcDbDocument currentDocument = null;


		[CommandMethod("sfd", CommandFlags.NoCheck | CommandFlags.NoPrefix)]
		public static void ScanFolder()
		{
			DirectoryScanner.Scan(@"C:\Users\mihan\Desktop\DwgCrawlTest");
		}

		/// <summary>
		/// Opens unscanned document
		/// </summary>
		[CommandMethod("grd", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void GetDocument()
		{
			DbMongo db = DbMongo.Instance;
			//While Get random dwg from database that not scanned
			var doc = db.GetNewRandomUnscannedDocument();

			if (doc != null)
			{
				// You will only see it once
				db.SetDocumentScanned(doc.FileId);

				currentDocument = new CrawlAcDbDocument(doc);
			}
		}

		/// <summary>
		/// Writes selection as a fragment to database
		/// </summary>
		[CommandMethod("fra", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Redraw)]
		public static void WriteFragment()
		{
			List<McObjectId> currentSelection = new List<McObjectId>(McObjectManager.SelectionSet.CurrentSelection);
			if (currentSelection.Count == 0)
				currentSelection = McObjectManager.SelectObjects("Select fragment").ToList();

			// каждый набор объектов со своим Guid группы
			var groupId = Guid.NewGuid().ToString();
			currentDocument.DumpEntities(currentSelection, groupId);
		}
	}
}
