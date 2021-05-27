using DwgDump.Data;
using DwgDump.Enitites;
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
		public static void GetRandomUnscannedDocument()
		{
			//While Get random dwg from database that not scanned
			var doc = DbMongo.Instance.GetNewRandomUnscannedDocument();

			if (doc != null)
			{
				// You will only see it once
				DbMongo.Instance.SetDocumentScanned(doc.FileId);

				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();
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
			CrawlAcDbDocument.DumpFragment(currentSelection, currentDocument.FileId, groupId);
		}

		/// <summary>
		/// Rescans saved fragments to update objects' fileds in database
		/// </summary>
		[CommandMethod("rescan", CommandFlags.NoCheck | CommandFlags.NoPrefix | CommandFlags.Session)]
		public static void Rescan()
		{
			foreach (CrawlDocument doc in DbMongo.Instance.GetAllScannedDocuments())
			{
				currentDocument = new CrawlAcDbDocument(doc);
				currentDocument.DoOpen();

				foreach (string objId in DbMongo.Instance.GetIdsFromDoc(doc.FileId))
				{
					// BUG: This won't work as McObjectIds are not persistent between sessions

					McObjectId id = new McObjectId(new Guid(objId));
					if (id.IsNull)
						continue;

					var obj = id.GetObject();
					if (obj != null)
					{
						var cent = Converters.From(id);

						DbMongo.Instance.UpdateObject(id.ToString(), Converters.Serialize(cent));
					}
				}

				currentDocument.Document.Close();
			}
		}
	}
}
